"""Pass/fail taxonomy for (factor, horizon) combos with failure hypotheses.

Reads the ``stage1_screen`` and ``walk_forward`` sections of an eval report
and explains, deterministically, WHY a combo failed.  The reasons are
hypotheses for the proposer agents -- every key maps to a checkable gate:

Stage-1 failure reasons (from the ``stage1_screen`` row):

* ``nan_by_design``       -- factor is ~all-NaN in the panel (or n_obs == 0
                             on every horizon).  Known real case: SSE cancel
                             decode is unreliable, so ``order_arrival_60s`` /
                             ``cancel_ratio_60s`` are NaN by design on SSE.
* ``below_ic_level``      -- |mean IC| under the per-horizon floor.
* ``below_noise_floor``   -- |mean IC| under the permutation noise floor.
* ``low_icir``            -- |ICIR| < 0.5 (signal-to-noise too poor).
* ``t_below_hurdle``      -- NW t under the Harvey-Liu hurdle.
* ``fdr_not_passed``      -- rejected by the BHY step-up at q <= 0.10.
* ``horizon_mismatch``    -- this factor peaks at another horizon and has
                             already decayed below half-peak here.
* ``cost_dominated_suspect`` -- short horizon (<= 60 s) with a thin signal
                             level; turnover costs plausibly exceed the gross
                             edge.  A hypothesis: confirm with the backtest.

Walk-forward failure reasons (dominant failed sub-gate across folds):

* ``oos_decay`` / ``oos_sign_flip`` / ``oos_t_low`` / ``oos_win_rate_low`` /
  ``oos_level_low`` -- the Stage-2 pristine-OOS sub-gates.
* ``no_walk_forward`` -- the date window produced no folds at all.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..eval.gating import GateConfig

__all__ = ["classify_outcomes", "REASON_ZH"]

#: a factor whose NaN rate reaches this is treated as NaN-by-design
NAN_BY_DESIGN_RATE = 0.999

#: walk-forward is considered passed when at least half the folds pass
WF_PASS_MAJORITY = 0.5

REASON_ZH: dict[str, str] = {
    "nan_by_design": (
        "因子在面板中几乎全为 NaN（或各 horizon 有效观测为 0），无法评估——"
        "已知情形：SSE 撤单/下单解码不可靠，order_arrival_60s 与 "
        "cancel_ratio_60s 在 SSE 上按设计为 NaN"
    ),
    "below_ic_level": "abs(mean IC) 低于该 horizon 的准入门槛",
    "below_noise_floor": "abs(mean IC) 低于置换检验噪声地板（信号不高于随机对齐）",
    "low_icir": "ICIR 过低（< 0.5，信噪比不足）",
    "t_below_hurdle": "Newey-West t 低于 Harvey-Liu 门槛（有效样本不足/方差过大）",
    "fdr_not_passed": "未通过 BHY-FDR 多重检验校正（q <= 0.10）",
    "horizon_mismatch": (
        "horizon 失配：该因子 abs(IC) 峰值在其他 horizon，"
        "本 horizon 已衰减至峰值一半以下"
    ),
    "cost_dominated_suspect": (
        "疑似成本主导：短 horizon（≤60s）信号幅度薄，换手成本大概率吃掉毛利，"
        "需回测确认"
    ),
    "no_walk_forward": "无 walk-forward 折结果（日期窗口不足以产生切分）",
    "oos_decay": "IS→OOS 保留率不足（< 0.5），疑似过拟合或机制衰减",
    "oos_sign_flip": "IS→OOS 符号翻转",
    "oos_t_low": "OOS 的 NW t 不足（< 2.0）",
    "oos_win_rate_low": "OOS 胜率不足（< 0.55）",
    "oos_level_low": "OOS abs(mean IC) 低于该 horizon 地板",
}

#: mapping walk-forward detail flag -> reason key
_WF_DETAIL_TO_REASON = (
    ("retention_ok", "oos_decay"),
    ("sign_ok", "oos_sign_flip"),
    ("oos_t_ok", "oos_t_low"),
    ("win_rate_ok", "oos_win_rate_low"),
    ("level_ok", "oos_level_low"),
)


def _f(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _stage1_index(eval_report: Mapping) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for row in eval_report.get("stage1_screen") or []:
        try:
            h = int(row.get("horizon_s"))
        except (TypeError, ValueError):
            continue
        out[(str(row.get("factor")), h)] = row
    return out


def _wf_index(eval_report: Mapping) -> dict[tuple[str, int], dict]:
    """Aggregate walk-forward folds per (factor, horizon)."""
    acc: dict[tuple[str, int], dict] = {}
    for row in eval_report.get("walk_forward") or []:
        try:
            h = int(row.get("horizon_s"))
        except (TypeError, ValueError):
            continue
        key = (str(row.get("factor")), h)
        slot = acc.setdefault(
            key, {"n_folds": 0, "n_passed": 0, "failed_gates": {}}
        )
        slot["n_folds"] += 1
        if bool(row.get("passed")):
            slot["n_passed"] += 1
        details = row.get("details") or {}
        for flag, reason in _WF_DETAIL_TO_REASON:
            if details.get(flag) is False:
                slot["failed_gates"][reason] = slot["failed_gates"].get(reason, 0) + 1
    for slot in acc.values():
        slot["pass_rate"] = (
            slot["n_passed"] / slot["n_folds"] if slot["n_folds"] else None
        )
    return acc


def _stats_index(eval_report: Mapping) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for row in eval_report.get("stats") or []:
        try:
            h = int(row.get("horizon_s"))
        except (TypeError, ValueError):
            continue
        out.setdefault(str(row.get("factor")), {})[h] = row
    return out


def _is_nan_by_design(
    factor: str,
    stats_by_factor: dict[str, dict[int, dict]],
    nan_rates: Mapping[str, float] | None,
    nan_rates_by_exchange: Mapping[str, Mapping[str, float]] | None,
) -> bool:
    if nan_rates and _f(nan_rates.get(factor)) >= NAN_BY_DESIGN_RATE:
        return True
    if nan_rates_by_exchange:
        for rates in nan_rates_by_exchange.values():
            if _f(rates.get(factor)) >= NAN_BY_DESIGN_RATE:
                return True
    rows = stats_by_factor.get(factor)
    if rows and all(int(r.get("n_obs") or 0) == 0 for r in rows.values()):
        return True
    return False


def _peak_horizon(
    stats_rows: dict[int, dict], horizons: Sequence[int]
) -> tuple[int | None, float]:
    peak_h: int | None = None
    peak_abs = -1.0
    for h in horizons:
        ic = _f(stats_rows.get(h, {}).get("mean_ic"))
        if math.isfinite(ic) and abs(ic) > peak_abs:
            peak_abs = abs(ic)
            peak_h = h
    return peak_h, peak_abs


def classify_outcomes(
    eval_report: Mapping,
    *,
    nan_rates: Mapping[str, float] | None = None,
    nan_rates_by_exchange: Mapping[str, Mapping[str, float]] | None = None,
    gate_cfg: GateConfig | None = None,
) -> list[dict]:
    """Deterministic pass/fail taxonomy over every evaluated combo.

    ``nan_rates`` / ``nan_rates_by_exchange`` are the panel NaN fractions
    from :func:`hft_autofactor.digest.panel_quality` (optional -- the
    taxonomy falls back to n_obs == 0 when no panel is available).
    """
    cfg = gate_cfg or GateConfig()
    stage1 = _stage1_index(eval_report)
    wf = _wf_index(eval_report)
    stats_by_factor = _stats_index(eval_report)

    horizons = sorted(
        {int(h) for _, h in stage1}
        | {int(h) for h in (eval_report.get("horizons_s") or [])}
    )

    combos: set[tuple[str, int]] = set(stage1) | set(wf)
    out: list[dict] = []
    for factor, h in sorted(combos, key=lambda k: (k[0], k[1])):
        s1 = stage1.get((factor, h))
        wf_slot = wf.get((factor, h))
        stage1_passed = bool(s1 and s1.get("passed"))
        if wf_slot is None:
            wf_pass_rate: float | None = None
            wf_passed = False
        else:
            wf_pass_rate = wf_slot["pass_rate"]
            wf_passed = bool(
                wf_pass_rate is not None and wf_pass_rate >= WF_PASS_MAJORITY
            )

        reasons: list[str] = []
        nan_designed = _is_nan_by_design(
            factor, stats_by_factor, nan_rates, nan_rates_by_exchange
        )

        if not stage1_passed:
            if nan_designed:
                # NaN-by-design needs no further speculation -- the factor
                # was never really evaluated on this data root.
                reasons.append("nan_by_design")
            elif s1 is not None:
                mean_ic = _f(s1.get("mean_ic"))
                abs_ic = abs(mean_ic) if math.isfinite(mean_ic) else 0.0
                floor = float(cfg.min_rank_ic.get(h, 0.02))
                icir = _f(s1.get("icir"))
                t_nw = _f(s1.get("t_stat_nw"))
                hurdle = _f(s1.get("t_hurdle_min"))
                noise = _f(s1.get("noise_floor"))
                if abs_ic < floor:
                    reasons.append("below_ic_level")
                if math.isfinite(noise) and abs_ic < noise:
                    reasons.append("below_noise_floor")
                if not (math.isfinite(icir) and abs(icir) >= cfg.min_icir):
                    reasons.append("low_icir")
                if not (math.isfinite(t_nw) and math.isfinite(hurdle)
                        and t_nw >= hurdle):
                    reasons.append("t_below_hurdle")
                if not s1.get("fdr_pass"):
                    reasons.append("fdr_not_passed")
                peak_h, peak_abs = _peak_horizon(
                    stats_by_factor.get(factor, {}), horizons
                )
                if (
                    peak_h is not None
                    and peak_h != h
                    and peak_abs > 0.0
                    and abs_ic < peak_abs / 2.0
                ):
                    reasons.append("horizon_mismatch")
                if h <= 60 and abs_ic < 2.0 * floor and "nan_by_design" not in reasons:
                    reasons.append("cost_dominated_suspect")

        if stage1_passed and not wf_passed:
            if wf_slot is None or wf_slot["n_folds"] == 0:
                reasons.append("no_walk_forward")
            else:
                failed = wf_slot["failed_gates"]
                if failed:
                    dominant = max(failed.items(), key=lambda kv: kv[1])[0]
                    # deterministic tie-break: fixed gate order
                    order = [r for _, r in _WF_DETAIL_TO_REASON]
                    best = sorted(
                        (k for k, v in failed.items() if v == failed[dominant]),
                        key=order.index,
                    )
                    reasons.extend(best[:1])
                else:  # pragma: no cover - defensive
                    reasons.append("oos_decay")

        reasons_zh = [REASON_ZH[r] for r in reasons]

        out.append(
            {
                "factor": factor,
                "horizon_s": h,
                "stage1_passed": stage1_passed,
                "wf_pass_rate": wf_pass_rate,
                "wf_passed": wf_passed,
                "combined_passed": bool(stage1_passed and wf_passed),
                "nan_by_design": bool(nan_designed),
                "failure_reasons": reasons,
                "reasons_zh": reasons_zh,
            }
        )
    return out
