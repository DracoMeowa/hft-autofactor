"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_pos_roll_range_300s: position of the OPEN anchor within the trailing-
300s mid range -- has the last five minutes explored above or below the
opening consensus?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100      # 100 x 3s rows = 300s rolling range window
EPS = 1e-12  # below this window range the position is undefined


def compute(part: pl.DataFrame) -> pl.Series:
    """(open - roll_min)/(roll_max - roll_min); warm-up / flat-window null.
    Values leave [0,1] when the whole window traded on one side of the
    open (extrapolated position -- by design)."""
    mid = pl.col("mid_px")
    hi = mid.rolling_max(window_size=W, min_samples=W)
    lo = mid.rolling_min(window_size=W, min_samples=W)
    rng = hi - lo
    out = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((pl.col("open_px") - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="open_pos_roll_range_300s",
    mechanism=(
        "Local exploration asymmetry around the anchor: over the last "
        "five minutes, has price been discovering territory ABOVE the "
        "opening consensus, BELOW it, or oscillating across it? When the "
        "open sits near the bottom of the recent window (value ~0 or "
        "below), every recent tick of exploration happened above the "
        "anchor -- the local battle was won upward, dips toward the open "
        "were bought, and the local drift regime is up (mirror below). "
        "An open mid-window marks two-sided churn around the anchor where "
        "reversion dominates. The 300s window recycles many times a day, "
        "so this tracks the CURRENT skirmish rather than the cumulative "
        "day shape (dev_from_open_bps) -- a mid-morning reversal shows up "
        "here minutes before it moves the whole-day statistics. Rolling "
        "extremes keep the value path-dependent every snapshot, unlike a "
        "cumulative-range construction that only moves on new records."
    ),
    info_set="mid_px, open_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 4 (where the open "
        "sits within the day range), implemented on the trailing 300s "
        "range for robustness; local-vs-cumulative split mirrors the "
        "round-2 mid_roll_range_pos_300s lesson (recycling windows track "
        "fast regime shifts the cumulative champion misses)."
    ),
    compute=compute,
)
