"""Explore-lane prototype spec (iter-003, etf-regime lens).

iopv_premium_z_600s: IOPV premium relative to its own trailing-600s
distribution.  Slow reference frame companion of iopv_premium_z_120s:
"unusual vs the last ten minutes", the frame the creation/redemption cycle
actually operates on.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 600s (200 x 3s rows) causal z-score window
Z_WINDOW = 200


def compute(part: pl.DataFrame) -> pl.Series:
    """Causal z-score of iopv_premium over 200 rows.

    Warm-up (< 200 rows) is null; constant trailing windows (std == 0) map
    to 0.0 (neutral) per the causal_zscore convention.
    """
    x = pl.col("iopv_premium")
    mean = x.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = x.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="iopv_premium_z_600s",
    mechanism=(
        "Regime-relative premium stretch on the arbitrage cycle's own time "
        "scale: ETF creation/redemption arbitrage is noticed within seconds "
        "but executes on minutes (basket construction, hedging, inventory "
        "limits), so a premium unusual versus the trailing TEN minutes is "
        "the object that separates exhausted episodes (already arbitraged "
        "into the recent norm) from live ones (still outside it). A "
        "premium z-scored high against 600s of history means the mispricing "
        "survived several arbitrage cycles -- either capacity-constrained "
        "(limits to arbitrage -> slow but eventual closure, reversion) or "
        "flow-driven and still being fed; in both readings the deviation "
        "from the slow norm is the risk-bearing state variable, and high "
        "values predict reversion toward IOPV at 60-900s horizons. "
        "Distinct from the 120s companion: the slow frame stays non-zero "
        "through sustained stretch episodes where the fast frame has "
        "re-normalized, so the two decorrelate exactly in the regime "
        "episodes this family targets."
    ),
    info_set="iopv_premium",
    inspiration=(
        "iter-001/iter-002 archive lessons: level/momentum forms of the "
        "premium died; relative-to-own-history transforms are the open "
        "lane (iter-003 etf-regime brief). Creation/redemption latency "
        "argues for a minutes-scale reference frame; limits to arbitrage "
        "(Shleifer & Vishny 1997) for why survived stretches eventually "
        "close."
    ),
    compute=compute,
)
