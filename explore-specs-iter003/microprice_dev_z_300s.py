"""Explore-lane prototype spec (iter-003, depth-book lens).

microprice_dev_z_300s: trailing-300s z-score of microprice_dev --
sustained queue-pressure REGIME vs own recent distribution.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def compute(part: pl.DataFrame) -> pl.Series:
    x = pl.col("microprice_dev")
    mean = x.rolling_mean(window_size=W, min_samples=W)
    std = x.rolling_std(window_size=W, min_samples=W)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="microprice_dev_z_300s",
    mechanism=(
        "Queue-pressure regime: microprice_dev z-scored against its own "
        "trailing-300s distribution. When the quantity-weighted touch price "
        "sits persistently above/below the mid RELATIVE TO ITS RECENT "
        "VARIABILITY, the touch has been systematically skewed to one side "
        "for a while - a sustained queue-positioning regime, not a transient "
        "event. Z-scoring against the trailing regime (rather than a raw "
        "level, which is the dead-end class) normalizes out the instrument's "
        "typical queue noise and flags only unusually committed one-sided "
        "queuing, which should carry into 60-900s drift."
    ),
    info_set="microprice_dev (library factor)",
    inspiration=(
        "iter-003 family brief: microprice leg; Stoikov (2018) micro-price; "
        "trailing-regime z convention from spread_z_300s / book_slope_z; "
        "level-vs-regime separation per the iter-001/002 meta-lesson."
    ),
    compute=compute,
)
