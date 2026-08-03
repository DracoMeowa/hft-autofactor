"""IC / RankIC metrics with serial-correlation-corrected inference.

Rank (Spearman) IC is the primary short-horizon metric: robust to the fat
tails of 3-second ETF returns.  Because 3s rows with overlapping 15s-900s
labels are heavily autocorrelated, t-statistics are computed on a Newey-West
/ Lo-adjusted effective sample size -- naive t would be inflated by roughly
an order of magnitude.

No scipy dependency: ranking and the normal CDF are implemented here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = [
    "ICStats",
    "spearman",
    "newey_west_n_eff",
    "rank_ic_time_series",
    "rank_ic_cross_section",
    "ic_stats",
    "label_column",
]


def label_column(horizon_s: int, label: str = "fwd_mid_ret") -> str:
    """Panel label column for a horizon, e.g. ``fwd_mid_ret_60s``."""
    return f"{label}_{horizon_s}s"


@dataclass
class ICStats:
    factor: str
    horizon_s: int
    n_obs: int
    mean_ic: float
    ic_std: float
    icir: float
    t_stat_nw: float
    n_eff: float
    win_rate: float


# --------------------------------------------------------------------- #
# ranking / correlation                                                 #
# --------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with average tie-breaking (scipy 'average' semantics)."""
    n = a.size
    sorter = np.argsort(a, kind="mergesort")
    sorted_a = a[sorter]
    obs = np.concatenate(([1], (sorted_a[1:] != sorted_a[:-1]).astype(np.int64)))
    nonzero = np.flatnonzero(np.concatenate((obs, [1])))
    counts = np.diff(nonzero)
    starts = nonzero[:-1]
    avg_rank = starts + (counts + 1) / 2.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = np.repeat(avg_rank, counts)
    return ranks


def spearman(x: np.ndarray | list, y: np.ndarray | list) -> float:
    """NaN-aware Spearman rank correlation on pairwise-complete observations.

    Returns NaN when fewer than 2 usable observations or either side has no
    dispersion after ranking (ties-only).
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise ValueError("spearman: shape mismatch")
    mask = np.isfinite(xa) & np.isfinite(ya)
    n = int(mask.sum())
    if n < 2:
        return float("nan")
    xr = _rankdata(xa[mask])
    yr = _rankdata(ya[mask])
    sx, sy = xr.std(), yr.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


# --------------------------------------------------------------------- #
# Newey-West / Lo effective sample size                                 #
# --------------------------------------------------------------------- #
def newey_west_n_eff(x: np.ndarray | list, max_lag: int | None = None) -> float:
    """Effective n of the mean of a serially-correlated series (Lo 2002).

    ``n_eff = n * gamma0 / (gamma0 + 2 * sum_k w_k gamma_k)`` with Bartlett
    weights ``w_k = 1 - k/(L+1)``.  Callers should pass
    ``max_lag = horizon_s // snapshot_period_s`` for overlapping-label
    series; the default heuristic is ``min(n-1, max(5, floor(sqrt(n))))``.
    Result is clamped to [1, n].
    """
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 3:
        return float(n)
    d = arr - arr.mean()
    gamma0 = float(d @ d) / n
    if gamma0 <= 0.0:
        return 1.0
    if max_lag is None:
        max_lag = min(n - 1, max(5, int(math.floor(math.sqrt(n)))))
    max_lag = max(0, min(max_lag, n - 1))
    denom = gamma0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        gamma_k = float(d[k:] @ d[:-k]) / n
        denom += 2.0 * w * gamma_k
    if denom <= 0.0:
        return 1.0
    n_eff = n * gamma0 / denom
    return float(min(n, max(1.0, n_eff)))


# --------------------------------------------------------------------- #
# per-group RankIC series                                               #
# --------------------------------------------------------------------- #
def _select_pair(
    panel: pl.DataFrame, factor: str, horizon_s: int, label: str
) -> pl.DataFrame:
    col_l = label_column(horizon_s, label)
    for col in (factor, col_l):
        if col not in panel.columns:
            raise KeyError(f"panel lacks column {col!r}; have {panel.columns}")
    return panel.select(
        [c for c in ("date", "instrument", "ts_ms") if c in panel.columns]
        + [factor, col_l]
    ).drop_nulls(subset=[factor, col_l])


def rank_ic_time_series(
    panel: pl.DataFrame,
    factor: str,
    horizon_s: int,
    label: str = "fwd_mid_ret",
) -> pl.DataFrame:
    """One Spearman IC per (instrument, date) over that day's rows.

    Returns columns ``date, instrument, ic, n`` (may include NaN ic for
    degenerate days, filtered downstream by :func:`ic_stats`).
    """
    col_l = label_column(horizon_s, label)
    df = _select_pair(panel, factor, horizon_s, label)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Utf8,
                "instrument": pl.Utf8,
                "ic": pl.Float64,
                "n": pl.UInt32,
            }
        )

    dates: list[str] = []
    insts: list[str] = []
    ics: list[float] = []
    ns: list[int] = []
    for part in df.partition_by(["date", "instrument"]):
        x = part[factor].to_numpy()
        y = part[col_l].to_numpy()
        dates.append(str(part["date"][0]))
        insts.append(str(part["instrument"][0]))
        ics.append(spearman(x, y))
        ns.append(len(part))
    return pl.DataFrame(
        {"date": dates, "instrument": insts, "ic": ics, "n": ns},
        schema={"date": pl.Utf8, "instrument": pl.Utf8, "ic": pl.Float64, "n": pl.UInt32},
    )


def rank_ic_cross_section(
    panel: pl.DataFrame,
    factor: str,
    horizon_s: int,
    *,
    min_instruments: int = 5,
    label: str = "fwd_mid_ret",
) -> pl.DataFrame:
    """One Spearman IC per (date, ts_ms) across instruments.

    Timestamps with fewer than ``min_instruments`` instruments are dropped
    (rank correlation across 2-3 ETFs is noise).  Returns columns
    ``date, ts_ms, ic, n_instruments``.
    """
    col_l = label_column(horizon_s, label)
    df = _select_pair(panel, factor, horizon_s, label)
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Utf8,
                "ts_ms": pl.Int64,
                "ic": pl.Float64,
                "n_instruments": pl.UInt32,
            }
        )

    dates: list[str] = []
    tss: list[int] = []
    ics: list[float] = []
    ns: list[int] = []
    for part in df.partition_by(["date", "ts_ms"]):
        if len(part) < min_instruments:
            continue
        dates.append(str(part["date"][0]))
        tss.append(int(part["ts_ms"][0]))
        ics.append(spearman(part[factor].to_numpy(), part[col_l].to_numpy()))
        ns.append(len(part))
    return pl.DataFrame(
        {"date": dates, "ts_ms": tss, "ic": ics, "n_instruments": ns},
        schema={
            "date": pl.Utf8,
            "ts_ms": pl.Int64,
            "ic": pl.Float64,
            "n_instruments": pl.UInt32,
        },
    )


def ic_stats(
    ic_df: pl.DataFrame,
    factor: str,
    horizon_s: int,
    *,
    ic_col: str = "ic",
    max_lag: int | None = None,
) -> ICStats:
    """Aggregate an IC series into :class:`ICStats` with NW-corrected t."""
    arr = ic_df[ic_col].to_numpy().astype(np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return ICStats(
            factor=factor,
            horizon_s=horizon_s,
            n_obs=0,
            mean_ic=float("nan"),
            ic_std=float("nan"),
            icir=float("nan"),
            t_stat_nw=float("nan"),
            n_eff=0.0,
            win_rate=float("nan"),
        )
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    icir = mean / std if std > 0.0 else float("nan")
    n_eff = newey_west_n_eff(arr, max_lag=max_lag)
    t_stat = mean / (std / math.sqrt(n_eff)) if std > 0.0 and n_eff > 0 else 0.0
    if mean > 0:
        win_rate = float((arr > 0).mean())
    elif mean < 0:
        win_rate = float((arr < 0).mean())
    else:
        win_rate = 0.0
    return ICStats(
        factor=factor,
        horizon_s=horizon_s,
        n_obs=n,
        mean_ic=mean,
        ic_std=std,
        icir=icir,
        t_stat_nw=float(t_stat),
        n_eff=n_eff,
        win_rate=win_rate,
    )
