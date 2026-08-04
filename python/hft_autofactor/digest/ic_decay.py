"""IC decay summary per factor and a coarse half-life estimate.

Consumes the ``stats`` list of the eval-stage JSON report (one entry per
factor x horizon with the :class:`~hft_autofactor.eval.ic.ICStats` fields;
NaNs are stored as ``null`` by the eval writer) and produces, per factor:

* the mean RankIC, NW t, ICIR, n_obs and win rate at every horizon;
* the peak horizon (smallest horizon achieving the max |mean IC|);
* a coarse half-life: the FIRST horizon strictly after the peak where
  |mean IC| drops below half the peak |mean IC|.  Only the decay side of
  the curve is considered -- a factor whose |IC| is still RISING toward its
  peak does not "halve" at shorter horizons.  ``None`` means the decay is
  slower than the longest evaluated horizon (long-lived signal).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..config import DEFAULT_HORIZONS_S

__all__ = ["decay_table"]


def _f(value: Any) -> float:
    """JSON null / missing -> NaN; otherwise float."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _horizons_of(eval_report: Mapping, horizons: Sequence[int] | None) -> list[int]:
    if horizons:
        return sorted(int(h) for h in horizons)
    rep_h = eval_report.get("horizons_s") or []
    if rep_h:
        return sorted(int(h) for h in rep_h)
    return list(DEFAULT_HORIZONS_S)


def decay_table(
    eval_report: Mapping,
    horizons: Sequence[int] | None = None,
) -> list[dict]:
    """Per-factor IC decay curve + half-life from an eval report dict.

    Returns one entry per factor (in first-appearance order)::

        {"factor": str,
         "horizons": {15: {"mean_ic", "t_stat_nw", "icir", "n_obs",
                           "win_rate"} or None, ...},
         "peak_horizon_s": int | None,
         "peak_abs_ic": float | None,
         "half_life_s": int | None}
    """
    horizon_list = _horizons_of(eval_report, horizons)
    stats = eval_report.get("stats") or []

    by_factor: dict[str, dict[int, dict]] = {}
    order: list[str] = []
    for row in stats:
        factor = str(row.get("factor"))
        try:
            h = int(row.get("horizon_s"))
        except (TypeError, ValueError):
            continue
        if factor not in by_factor:
            by_factor[factor] = {}
            order.append(factor)
        by_factor[factor][h] = {
            "mean_ic": row.get("mean_ic"),
            "t_stat_nw": row.get("t_stat_nw"),
            "icir": row.get("icir"),
            "n_obs": row.get("n_obs") or 0,
            "win_rate": row.get("win_rate"),
        }

    out: list[dict] = []
    for factor in order:
        per_h = by_factor[factor]
        entry_horizons: dict[int, dict | None] = {}
        for h in horizon_list:
            entry_horizons[h] = per_h.get(h)

        # peak over the finite mean ICs
        peak_h: int | None = None
        peak_abs = -1.0
        for h in horizon_list:
            cell = per_h.get(h)
            if not cell:
                continue
            ic = _f(cell.get("mean_ic"))
            if math.isfinite(ic) and abs(ic) > peak_abs:
                peak_abs = abs(ic)
                peak_h = h
        if peak_h is None:
            out.append(
                {
                    "factor": factor,
                    "horizons": entry_horizons,
                    "peak_horizon_s": None,
                    "peak_abs_ic": None,
                    "half_life_s": None,
                }
            )
            continue

        half_life: int | None = None
        threshold = peak_abs / 2.0
        for h in horizon_list:
            if h <= peak_h:
                continue  # decay side only
            cell = per_h.get(h)
            if not cell:
                continue
            ic = _f(cell.get("mean_ic"))
            if math.isfinite(ic) and abs(ic) < threshold:
                half_life = h
                break

        out.append(
            {
                "factor": factor,
                "horizons": entry_horizons,
                "peak_horizon_s": peak_h,
                "peak_abs_ic": peak_abs,
                "half_life_s": half_life,
            }
        )
    return out
