"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

range_width_delta_120s: speed of intraday range expansion -- the 120s change
of (high_px - low_px) / mid_px. Information-arrival rate in range units.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_DIFF = 40  # 40 x 3s rows = 120s


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing-120s change of the day-range ratio; first 40 rows null."""
    width = (pl.col("high_px") - pl.col("low_px")) / pl.col("mid_px")
    return part.select(width.diff(W_DIFF).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_width_delta_120s",
    mechanism=(
        "Range expansion speed measures the rate of information arrival: the "
        "day's high/low only move when price explores beyond every earlier "
        "print, so a fast-widening (high-low) envelope flags an active "
        "price-discovery regime in which new information is being "
        "incorporated right now. Volatility and information flow cluster in "
        "time (attention cascades, stop runs, market-maker re-hedging), so a "
        "rapidly expanding envelope predicts that the current exploration "
        "persists and drift continues over the next minutes, while a "
        "stalling envelope marks exhaustion/consolidation with smaller "
        "future moves. The diff form isolates the CHANGE of regime activity, "
        "not the slowly growing level that round 1 showed is dead."
    ),
    info_set="high_px, low_px, mid_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 1 (range expansion "
        "speed): mid_day_range_pos was the round-1 all-horizon champion with "
        "max |rho| 0.41, making the day-range family a new signal source; "
        "this spec expands it along the expansion-speed axis. Volatility "
        "clustering (Engle 1982; Bollerslev 1986) applied to the intraday "
        "extreme envelope."
    ),
    compute=compute,
)
