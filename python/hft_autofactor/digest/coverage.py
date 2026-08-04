"""Panel information-dimension coverage and opportunity hints.

The panel carries several orthogonal information dimensions; the 12 library
factors do not exploit all of them.  This module keeps the (curated)
factor -> dimension map, reports which dimensions are uncovered or thinly
covered, ranks horizons by cross-factor strength (the weakest horizons are
also opportunity hints), and attaches curated research directions for the
proposer agents.

Dimension map rationale (v1 library):

* depth       -- oir, wdi, book_slope (level quantities / book shape);
                 microprice_dev also uses top-of-book quantities.
* quote       -- quoted_spread_ticks, microprice_dev (top-of-book prices);
                 rv_60s / rv_300s (log-mid return history).
* flow        -- ofi_60s, trade_imbalance_60s, order_arrival_60s,
                 cancel_ratio_60s (tick-stream aggregates).
* iopv        -- iopv_premium (NAV vs price).
* time_of_day -- NOTHING.  ts_ms never enters a factor; seasonality is the
                 biggest structural gap in the v1 library.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

__all__ = [
    "DIMENSIONS",
    "DIMENSION_ZH",
    "FACTOR_DIMENSIONS",
    "OPPORTUNITY_HINTS_ZH",
    "coverage_report",
]

DIMENSIONS: tuple[str, ...] = ("depth", "quote", "flow", "iopv", "time_of_day")

DIMENSION_ZH: dict[str, str] = {
    "depth": "深度（盘口量）",
    "quote": "报价（价格/价差/中间价）",
    "flow": "订单流（逐笔）",
    "iopv": "IOPV（净值）",
    "time_of_day": "日内时段",
}

#: factor -> dimensions it exploits (many-to-many is allowed)
FACTOR_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "quoted_spread_ticks": ("quote",),
    "microprice_dev": ("quote", "depth"),
    "oir": ("depth",),
    "wdi": ("depth",),
    "book_slope": ("depth",),
    "iopv_premium": ("iopv",),
    "rv_60s": ("quote",),
    "rv_300s": ("quote",),
    "ofi_60s": ("flow",),
    "trade_imbalance_60s": ("flow",),
    "order_arrival_60s": ("flow",),
    "cancel_ratio_60s": ("flow",),
}

#: curated research directions per dimension (hypotheses for proposers).
#: Written neutrally (what is explorable / still missing) so they stay valid
#: whether the dimension is a total gap or only thinly covered.
OPPORTUNITY_HINTS_ZH: dict[str, str] = {
    "time_of_day": (
        "ts_ms 从未进入任何因子。可探索：开盘/收盘竞价效应、午休前后效应、"
        "日内 U 形波动/流动性归一化、用时段条件化其它因子"
    ),
    "depth": (
        "盘口量维度可深挖：成交后深度恢复速度（resiliency）、深度变化率、"
        "队列位置、大单深度占比；现有因子多为静态失衡量"
    ),
    "flow": (
        "逐笔维度可深挖：VPIN（流毒性）、Kyle λ（价格冲击）、"
        "成交间隔（trade duration）、大单占比；现有因子多为计数/量失衡"
    ),
    "quote": (
        "报价维度可深挖：价差持续性（spread duration）、报价更新强度、"
        "连续同向报价跳（quote run）、微观价偏离的动态化"
    ),
    "iopv": (
        "IOPV 维度可深挖：溢价动量/变化率、溢价×订单流交互、"
        "溢价回归速度（折溢价套利节奏）；现有仅有溢价水平值"
    ),
}


def coverage_report(
    factors_present: Sequence[str],
    decay_rows: Sequence[Mapping] = (),
    stage1_rows: Sequence[Mapping] = (),
) -> dict:
    """Dimension coverage + weakest horizons + opportunity hints.

    * ``factors_present``: factors actually evaluated (eval report order).
    * ``decay_rows``: output of :func:`ic_decay.decay_table` (optional, for
      the weakest-horizon ranking by mean |IC|).
    * ``stage1_rows``: the eval report ``stage1_screen`` list (optional, for
      per-horizon pass counts).
    """
    factors_present = list(dict.fromkeys(factors_present))
    unmapped = [f for f in factors_present if f not in FACTOR_DIMENSIONS]

    coverage: dict[str, dict] = {}
    for dim in DIMENSIONS:
        covered_by = [
            f
            for f in factors_present
            if dim in FACTOR_DIMENSIONS.get(f, ())
        ]
        coverage[dim] = {
            "covered_by": covered_by,
            "covered": bool(covered_by),
            "thin": len(covered_by) == 1,
        }

    gaps = [dim for dim in DIMENSIONS if not coverage[dim]["covered"]]
    thin = [dim for dim in DIMENSIONS if coverage[dim]["thin"]]

    hints = [{"dimension": dim, "hint_zh": OPPORTUNITY_HINTS_ZH[dim]}
             for dim in gaps]
    hints += [{"dimension": dim, "hint_zh": OPPORTUNITY_HINTS_ZH[dim]}
              for dim in thin if dim not in gaps]

    # ---- weakest horizons -------------------------------------------- #
    horizon_ic: dict[int, list[float]] = {}
    for row in decay_rows:
        for h, cell in (row.get("horizons") or {}).items():
            if not cell:
                continue
            ic = cell.get("mean_ic")
            if ic is None:
                continue
            try:
                ic = float(ic)
            except (TypeError, ValueError):
                continue
            if math.isfinite(ic):
                horizon_ic.setdefault(int(h), []).append(abs(ic))

    pass_counts: dict[int, int] = {}
    combo_counts: dict[int, int] = {}
    for row in stage1_rows:
        try:
            h = int(row.get("horizon_s"))
        except (TypeError, ValueError):
            continue
        combo_counts[h] = combo_counts.get(h, 0) + 1
        if row.get("passed"):
            pass_counts[h] = pass_counts.get(h, 0) + 1

    weakest: list[dict] = []
    for h in sorted(set(horizon_ic) | set(combo_counts)):
        ics = horizon_ic.get(h, [])
        weakest.append(
            {
                "horizon_s": h,
                "mean_abs_ic": (sum(ics) / len(ics)) if ics else None,
                "n_factors_with_ic": len(ics),
                "stage1_pass": pass_counts.get(h, 0),
                "stage1_combos": combo_counts.get(h, 0),
            }
        )

    def _rank_key(w: dict):
        mic = w["mean_abs_ic"]
        return (mic if mic is not None else float("inf"), w["stage1_pass"])

    weakest.sort(key=_rank_key)

    return {
        "factor_dimension_map": {
            f: list(FACTOR_DIMENSIONS.get(f, ())) for f in factors_present
        },
        "unmapped_factors": unmapped,
        "dimension_coverage": {
            dim: {
                "zh": DIMENSION_ZH[dim],
                **coverage[dim],
            }
            for dim in DIMENSIONS
        },
        "gaps": gaps,
        "thin_dimensions": thin,
        "opportunity_hints": hints,
        "weakest_horizons": weakest,
    }
