"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

mid_roll_range_pos_300s: position of mid within the TRAILING 300s high-low
range (100 rows). Short-horizon sibling of the day-cumulative champion
mid_day_range_pos: local battle range instead of the whole-day anchor.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100    # 100 x 3s rows = 300s rolling window
EPS = 1e-12


def compute(part: pl.DataFrame) -> pl.Series:
    """(mid - roll_min)/(roll_max - roll_min); null in warm-up/flat windows."""
    mid = pl.col("mid_px")
    rmax = mid.rolling_max(window_size=W, min_samples=W)
    rmin = mid.rolling_min(window_size=W, min_samples=W)
    rng = rmax - rmin
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - rmin) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(pos.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="mid_roll_range_pos_300s",
    mechanism=(
        "Position of mid within the trailing-300s high-low range: the LOCAL "
        "version of range-position anchoring. The round-1 champion "
        "mid_day_range_pos proved that position inside the day's envelope "
        "encodes anchoring/stop-zone proximity with negative IC at all five "
        "horizons; but intraday scalpers and execution algos defend the "
        "boundaries of the RECENT battle range, which recycle many times a "
        "day. A mid near the top of the last 5 minutes' range sits just "
        "under fresh supply and just-printed highs (rejection or breakout "
        "fuel on a much faster clock), near the bottom just over fresh "
        "demand. Because the rolling window drops stale extremes, this "
        "factor tracks fast anchor shifts and decorrelates from the "
        "whole-day cumulative champion."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 3 (rolling-window "
        "range position, 100/200-row windows); direct expansion of the "
        "round-1 all-horizon champion mid_day_range_pos (anchoring on "
        "intraday extremes, Tversky & Kahneman 1974) to the short-anchor "
        "scale."
    ),
    compute=compute,
)
