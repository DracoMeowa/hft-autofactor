"""Explore-lane prototype spec (iter-003, price-vol family).

mid_day_range_pos: position of the current mid within the intraday
(cumulative) high-low range.  0 = at the day's low, 1 = at the day's high.
Anchoring / stop-zone proximity.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: below this range the position is undefined (flat day so far)
EPS = 1e-12


def compute(part: pl.DataFrame) -> pl.Series:
    """(mid - cum_min)/(cum_max - cum_min); null while the range is ~0."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(pos.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="mid_day_range_pos",
    mechanism=(
        "Position within the intraday range encodes anchoring and stop-zone "
        "proximity. Traders anchor on the day's high/low and cluster stop "
        "orders just beyond them, so a mid near the top of the range (value "
        "~1) sits next to a zone of resting buy-stops (breakout fuel) and "
        "overhead resistance, while a mid near the bottom (~0) sits next to "
        "sell-stops and support. Approaching an extreme predicts either "
        "continuation through the stop cluster or rejection off the level, "
        "and the exact rank position disambiguates where in the day's "
        "battle the current price sits -- information absent from pure "
        "momentum. Bounded [0,1], direction-free in level, ranked."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 price-vol family brief seed idea 13 (intraday-range "
        "position via causal cum_max/cum_min). Anchoring on round/intraday "
        "extremes (Tversky & Kahneman 1974); stop-hunting and breakout "
        "mechanics around prior high/low."
    ),
    compute=compute,
)
