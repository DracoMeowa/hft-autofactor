"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

roll_range_width_z_300s: z-score of the trailing-300s LOCAL range width
((roll_max - roll_min)/mid) against its own trailing-300s history.
Squeeze/release trigger on the stationary local envelope.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_RANGE = 100  # 100 x 3s rows = 300s local range window
W_Z = 100      # 100 x 3s rows = 300s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(local_range_width); constant windows -> 0.0; warm-up null."""
    mid = pl.col("mid_px")
    rmax = mid.rolling_max(window_size=W_RANGE, min_samples=W_RANGE)
    rmin = mid.rolling_min(window_size=W_RANGE, min_samples=W_RANGE)
    local_w = (rmax - rmin) / mid
    mean = local_w.rolling_mean(window_size=W_Z, min_samples=W_Z)
    std = local_w.rolling_std(window_size=W_Z, min_samples=W_Z)
    z = (local_w - mean) / std
    out = pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(z)
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="roll_range_width_z_300s",
    mechanism=(
        "Local squeeze/release trigger: the trailing-300s range width (as a "
        "fraction of price), z-scored against its own trailing-300s "
        "history. Falling z = the local envelope is CONTRACTING versus its "
        "recent self: a squeeze in which inventory and resting orders coil, "
        "compression that historically resolves in an abrupt directional "
        "move; rising z = the local envelope is exploding, and volatility "
        "clustering implies the burst (and its direction) persists in the "
        "near term. Distinct from the day-cumulative width z: the local "
        "window is a STATIONARY volatility state that catches "
        "compression/release cycles many times per day, whereas the "
        "cumulative envelope mostly remembers the morning."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief directions 3+7 (rolling range "
        "functionals, width-state z); squeeze mechanics and volatility "
        "clustering (Engle 1982; Bollerslev 1986), applied to the range "
        "envelope instead of squared returns (the RV-z variants died IS "
        "in round 1; the envelope measures extremum exploration, not path "
        "chop)."
    ),
    compute=compute,
)
