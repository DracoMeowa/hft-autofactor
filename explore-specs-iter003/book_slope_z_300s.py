"""Explore-lane prototype spec (iter-003, depth-book lens).

book_slope_z_300s: causal trailing-300s z-score of the engine
book_slope -- book SHAPE state relative to its own recent distribution.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def compute(part: pl.DataFrame) -> pl.Series:
    x = pl.col("book_slope")
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
    name="book_slope_z_300s",
    mechanism=(
        "Book-shape regime state: the engine book_slope (price-depth profile "
        "steepness/asymmetry) z-scored against its own trailing-300s "
        "distribution. Z-scoring separates a genuinely unusual shape REGIME "
        "from the instrument's baseline shape and from intraday "
        "non-stationarity. An unusually steep one-sided profile means "
        "limit-order traders are committing size asymmetrically far into the "
        "book - a sustained directional positioning state that decays more "
        "slowly than touch imbalances and should condition 300-900s drift "
        "direction. Constant trailing windows map to 0 (neutral), warm-up "
        "rows null."
    ),
    info_set="book_slope (library factor)",
    inspiration=(
        "iter-003 family brief: book-shape state interaction with the live "
        "depth-momentum dimension; Potters & Bouchaud (2003) statistical "
        "structure of limit-order book shape; distinct from dead "
        "vol_adj_slope (slope/vol LEVEL ratio): this is trailing-regime "
        "z-state, which the meta-lesson does not rule out."
    ),
    compute=compute,
)
