"""Pre-screen for prototypes: RankIC/NW inference, library dedup, IS/OOS.

A cheap gate between "prototype computes" and the full Stage-4 evaluation:

* per-horizon RankIC with Newey-West corrected t (reuses ``eval/ic.py``,
  ``max_lag = horizon // 3`` for the overlapping 3s-grid labels);
* duplication gate: Spearman correlation against the library factor
  columns, ``max |corr| > max_abs_corr (0.85) => duplicate => reject``.
  The dedup universe defaults to the 12 canonical library factors and is
  EXTENSIBLE per call (``library_factors=``): once wishlist columns are
  materialized into the panel, pass ``PANEL_FACTORS`` (or an explicit name
  list) so new candidates are also deduped against them;
* IS/OOS date split with a >= 1-day purge (reuses ``eval/splits.py``):
  purged anchored walk-forward folds, last fold as the canonical split,
  retention via ``is_oos_retention``.

Every screened (prototype, horizon) is appended to the shared trial ledger
BEFORE thresholds are read -- exploration trials count against the zoo's
honest N exactly like Stage-1 candidates.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from ..config import PipelineConfig
from ..eval.gating import TrialLedger
from ..eval.ic import ic_stats, label_column, rank_ic_time_series, spearman
from ..eval.splits import is_oos_retention, purged_day_splits
from ..ingest import DEFAULT_FACTORS, factor_columns
from .layout import screen_report_path, sanitize_for_json
from .registry import Prototype
from .runner import load_explore_panel

__all__ = [
    "PANEL_FACTORS",
    "ScreenConfig",
    "ScreenReport",
    "library_correlations",
    "screen_prototype",
]


class _PanelFactorsSentinel:
    """Sentinel for ``library_factors``: dedup against EVERY factor column
    present in the loaded panel (base/label/channel columns excluded, the
    prototype's own column excluded inside :func:`library_correlations`).
    Use once materialized wishlist columns extend the panel beyond the 12
    canonical library factors."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "PANEL_FACTORS"


#: Pass as ``library_factors`` to dedup against all panel factor columns.
PANEL_FACTORS = _PanelFactorsSentinel()


@dataclass
class ScreenConfig:
    max_abs_corr: float = 0.85     # > this vs any library factor => duplicate
    min_is_t: float = 2.0          # NW t on the IS block (loose pre-screen bar)
    min_oos_t: float = 2.0         # NW t on the pristine OOS block
    min_retention: float = 0.5     # |OOS IC| / |IS IC| with same sign
    embargo_days: int = 1          # purged days between train end and test start
    n_test_days: int = 5           # OOS block size of the walk-forward folds
    max_corr_obs: int = 200_000    # deterministic subsample cap for correlations


@dataclass
class ScreenReport:
    prototype: str
    status: str  # "ok" | "rejected_duplicate" | "insufficient_data" | "failed"
    passed: bool
    reasons: list[str] = field(default_factory=list)
    split: dict = field(default_factory=dict)
    duplicate_check: dict = field(default_factory=dict)
    horizons: list[dict] = field(default_factory=list)
    report_path: Path | None = None


def _stride_sample(n: int, max_obs: int) -> np.ndarray | None:
    """Deterministic even-stride subsample index (None => take all)."""
    if max_obs <= 0 or n <= max_obs:
        return None
    return np.unique(
        np.round(np.linspace(0.0, n - 1, max_obs)).astype(np.int64)
    )


def library_correlations(
    panel: pl.DataFrame,
    column: str,
    *,
    library_factors: Sequence[str] = DEFAULT_FACTORS,
    max_obs: int = 200_000,
) -> dict[str, float]:
    """Pooled pairwise-complete Spearman corr of ``column`` vs library factors.

    Labels are never involved, so pooling over all dates leaks nothing.
    Rows beyond ``max_obs`` are reduced by a deterministic even stride.
    Only library columns present in the panel are scored.
    """
    out: dict[str, float] = {}
    for f in library_factors:
        if f == column or f not in panel.columns:
            continue
        pair = (
            panel.select([column, f])
            .drop_nulls(subset=[column, f])
        )
        n = pair.height
        if n < 2:
            out[f] = float("nan")
            continue
        idx = _stride_sample(n, max_obs)
        x = pair[column].to_numpy()
        y = pair[f].to_numpy()
        if idx is not None:
            x, y = x[idx], y[idx]
        out[f] = spearman(x, y)
    return out


def _filter_dates(panel: pl.DataFrame, dates: Sequence[str]) -> pl.DataFrame:
    return panel.filter(pl.col("date").is_in(list(dates)))


def screen_prototype(
    cfg: PipelineConfig,
    proto: Prototype,
    dates: Sequence[str],
    *,
    horizons: Sequence[int] | None = None,
    screen_cfg: ScreenConfig | None = None,
    ledger: TrialLedger | None = None,
    library_factors: Sequence[str] | _PanelFactorsSentinel | None = None,
) -> ScreenReport:
    """Run the full pre-screen for one prototype over ``dates``.

    Requires the prototype's augmented partitions (``run_prototype`` first).
    Writes ``reports/screen_{name}_{stamp}.json`` (+ ``.csv`` horizon table)
    under the explore root and returns the :class:`ScreenReport`.

    ``library_factors`` sets the dedup universe: ``None`` (default) = the 12
    canonical library factors (unchanged historical behavior);
    :data:`PANEL_FACTORS` = every factor column present in the loaded panel
    (use once materialized wishlist columns extend the panel); or an explicit
    list of column names (names absent from the panel are skipped). The
    resolved list is recorded in the report's ``duplicate_check`` for audit.
    """
    sc = screen_cfg or ScreenConfig()
    dates_sorted = sorted(set(dates))
    if not dates_sorted:
        raise ValueError("screen_prototype: no dates given")

    panel = load_explore_panel(cfg, proto.name, dates_sorted)
    if proto.name not in panel.columns:
        raise ValueError(
            f"explore partitions lack prototype column {proto.name!r}"
        )

    horizons_eff = [
        int(h)
        for h in (horizons or cfg.horizons_s)
        if label_column(int(h)) in panel.columns
    ]

    # ---------------- dedup vs the library factors -------------------- #
    if library_factors is PANEL_FACTORS:
        lib = factor_columns(panel.columns)
    elif library_factors is None:
        lib = list(DEFAULT_FACTORS)
    else:
        lib = list(library_factors)
    corrs = library_correlations(
        panel, proto.name, library_factors=lib, max_obs=sc.max_corr_obs
    )
    abs_corrs = {
        f: (abs(v) if np.isfinite(v) else -1.0) for f, v in corrs.items()
    }
    if abs_corrs:
        worst = max(abs_corrs, key=lambda f: abs_corrs[f])
        max_abs_corr = abs_corrs[worst]
    else:
        worst, max_abs_corr = None, float("nan")
    duplicated = bool(np.isfinite(max_abs_corr) and max_abs_corr > sc.max_abs_corr)

    # ---------------- purged IS/OOS split ------------------------------ #
    splits = purged_day_splits(
        dates_sorted,
        n_test_days=min(sc.n_test_days, max(1, len(dates_sorted) - 2)),
        mode="anchored",
        embargo_days=sc.embargo_days,
    )
    ledger = ledger or TrialLedger(cfg.reports_dir / "trial_ledger.jsonl")

    reasons: list[str] = []
    if duplicated:
        reasons.append(
            f"duplicate of library factor {worst!r} "
            f"(max |corr| = {max_abs_corr:.3f} > {sc.max_abs_corr})"
        )
    if not splits:
        report = ScreenReport(
            prototype=proto.name,
            status="insufficient_data",
            passed=False,
            reasons=reasons + [
                f"no purged IS/OOS split possible for {len(dates_sorted)} "
                f"dates (n_test_days={sc.n_test_days}, embargo={sc.embargo_days})"
            ],
            duplicate_check={
                "max_abs_corr": max_abs_corr,
                "library_factor": worst,
                "threshold": sc.max_abs_corr,
                "duplicated": duplicated,
                "library_factors": lib,
                "corrs": corrs,
            },
        )
        _write_screen_report(cfg, proto, dates_sorted, report)
        return report

    split = splits[-1]  # canonical: anchored walk-forward, last fold
    is_panel = _filter_dates(panel, split.train_dates)
    oos_panel = _filter_dates(panel, split.test_dates)

    # ---------------- per-horizon RankIC screen ------------------------ #
    horizon_rows: list[dict] = []
    for h in horizons_eff:
        is_st = ic_stats(
            rank_ic_time_series(is_panel, proto.name, h),
            proto.name, h, max_lag=h // 3,
        )
        oos_st = ic_stats(
            rank_ic_time_series(oos_panel, proto.name, h),
            proto.name, h, max_lag=h // 3,
        )
        # honest-N bookkeeping: every screened (proto, horizon) is a trial,
        # logged BEFORE thresholds are read
        ledger.log(
            factor=proto.name,
            horizon_s=h,
            params={"stage": "explore_screen"},
            stage="explore_screen",
            metrics={
                "is_mean_ic": is_st.mean_ic,
                "is_t_stat_nw": is_st.t_stat_nw,
                "oos_mean_ic": oos_st.mean_ic,
                "oos_t_stat_nw": oos_st.t_stat_nw,
            },
        )
        retention_ok = is_oos_retention(
            is_st.mean_ic, oos_st.mean_ic, min_retention=sc.min_retention
        )
        is_t_ok = (
            np.isfinite(is_st.t_stat_nw) and abs(is_st.t_stat_nw) >= sc.min_is_t
        )
        oos_t_ok = (
            np.isfinite(oos_st.t_stat_nw) and abs(oos_st.t_stat_nw) >= sc.min_oos_t
        )
        horizon_rows.append(
            {
                "horizon_s": h,
                "is_mean_ic": is_st.mean_ic,
                "is_icir": is_st.icir,
                "is_t_stat_nw": is_st.t_stat_nw,
                "is_n_obs": is_st.n_obs,
                "oos_mean_ic": oos_st.mean_ic,
                "oos_icir": oos_st.icir,
                "oos_t_stat_nw": oos_st.t_stat_nw,
                "oos_n_obs": oos_st.n_obs,
                "retention": (
                    abs(oos_st.mean_ic) / abs(is_st.mean_ic)
                    if np.isfinite(is_st.mean_ic) and is_st.mean_ic != 0.0
                    else float("nan")
                ),
                "retention_ok": retention_ok,
                "is_t_ok": bool(is_t_ok),
                "oos_t_ok": bool(oos_t_ok),
                "passed": bool(retention_ok and is_t_ok and oos_t_ok),
            }
        )

    any_horizon_passed = any(r["passed"] for r in horizon_rows)
    if not horizons_eff:
        reasons.append("no label columns present for any configured horizon")
    elif not any_horizon_passed:
        reasons.append(
            "no horizon passed (need IS |t| >= "
            f"{sc.min_is_t}, OOS |t| >= {sc.min_oos_t}, retention >= "
            f"{sc.min_retention} with same sign)"
        )

    passed = (not duplicated) and any_horizon_passed
    if duplicated and any_horizon_passed:
        reasons.append("signal present but already covered by the library")

    status = "rejected_duplicate" if duplicated else "ok"
    report = ScreenReport(
        prototype=proto.name,
        status=status,
        passed=passed,
        reasons=reasons,
        split={
            "train_dates": list(split.train_dates),
            "test_dates": list(split.test_dates),
            "embargo_days": sc.embargo_days,
            "mode": "anchored",
        },
        duplicate_check={
            "max_abs_corr": max_abs_corr,
            "library_factor": worst,
            "threshold": sc.max_abs_corr,
            "duplicated": duplicated,
            "library_factors": lib,
            "corrs": corrs,
        },
        horizons=horizon_rows,
    )
    _write_screen_report(cfg, proto, dates_sorted, report)
    return report


def _write_screen_report(
    cfg: PipelineConfig, proto: Prototype, dates: list[str], report: ScreenReport
) -> Path:
    path = screen_report_path(cfg, proto.name, dates)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prototype": proto.metadata_dict(),
        "status": report.status,
        "passed": report.passed,
        "reasons": report.reasons,
        "dates": dates,
        "split": report.split,
        "duplicate_check": report.duplicate_check,
        "horizons": report.horizons,
    }
    path.write_text(
        json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if report.horizons:
        pl.DataFrame(report.horizons).write_csv(path.with_suffix(".csv"))
    report.report_path = path
    return path
