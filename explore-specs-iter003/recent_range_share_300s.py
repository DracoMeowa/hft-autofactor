"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

recent_range_share_300s: fraction of the day's total (cumulative) mid-range
that is contained in the trailing 300s mid-range. Volatility recency: is the
day's exploration happening NOW or did it already happen?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100    # 100 x 3s rows = 300s trailing window
EPS = 1e-12


def compute(part: pl.DataFrame) -> pl.Series:
    """rolling_range_300s / cumulative_day_range; warm-up rows null."""
    mid = pl.col("mid_px")
    rmax = mid.rolling_max(window_size=W, min_samples=W)
    rmin = mid.rolling_min(window_size=W, min_samples=W)
    recent = rmax - rmin
    day = mid.cum_max() - mid.cum_min()
    share = (
        pl.when(recent.is_not_null() & (day > EPS))
        .then(recent / day)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(share.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="recent_range_share_300s",
    mechanism=(
        "Volatility recency: the share of the day's total range explored "
        "within the last five minutes, bounded [0,1]. A high share means "
        "the day's boundaries are being made RIGHT NOW -- fresh discovery "
        "regime in which the most recent information is still driving and "
        "drift tends to persist; a low share means the market has gone "
        "quiet inside an envelope built earlier -- consolidation in which "
        "the prior impulse has decayed and reversion toward mid-range "
        "dominates. Unlike absolute range width, this separates WHEN the "
        "range was made from how big it is: two days with identical width "
        "can differ in active-now vs done-for-the-day, and their next-move "
        "distributions differ accordingly. Scale-free by construction."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief directions 3+7 (rolling-window "
        "range functionals on the champion's cumulative envelope); "
        "recency-weighting of volatility states as in the round-1 lesson "
        "that deltas/states beat slow levels."
    ),
    compute=compute,
)
