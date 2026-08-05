"""Explore-lane prototype spec (iter-003, depth-book lens).

depth_ratio_5to1_z: trailing-300s z-score of (5-level total size) /
(top-of-book size) -- size stacked BEHIND the touch.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def compute(part: pl.DataFrame) -> pl.Series:
    b = pl.col("depth_bid5").cast(pl.Float64)
    a = pl.col("depth_ask5").cast(pl.Float64)
    bq = pl.col("bid1_qty").cast(pl.Float64)
    aq = pl.col("ask1_qty").cast(pl.Float64)
    deep = b + a
    top = bq + aq
    ratio = (
        pl.when(top > 0.0)
        .then(deep / top)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    mean = ratio.rolling_mean(window_size=W, min_samples=W)
    std = ratio.rolling_std(window_size=W, min_samples=W)
    z = (ratio - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="depth_ratio_5to1_z",
    mechanism=(
        "Hidden layering behind the touch: the ratio of total 5-level size "
        "to top-of-book size, z-scored against its trailing-300s "
        "distribution. A high ratio means thick size is stacked BEHIND the "
        "displayed queue - patient, iceberg-style layering where visible "
        "queue consumption gets replenished from reserves (absorption "
        "regime, low effective impact). A low ratio means a top-heavy, "
        "fragile book where the displayed size is all there is (high impact "
        "regime, consumption continues). Being scale-free, this layering "
        "dimension is distinct from total-thickness z (which mixes in the "
        "overall size regime) and from imbalance ratios (which ignore "
        "distribution across levels)."
    ),
    info_set="depth_bid5, depth_ask5, bid1_qty, ask1_qty",
    inspiration=(
        "iter-003 family brief: size stacked behind the touch; iceberg/"
        "undisplayed reserves (Buti & Rindi 2013); book-shape distribution "
        "across levels (Zovko & Farmer 2002); trailing-regime z convention."
    ),
    compute=compute,
)
