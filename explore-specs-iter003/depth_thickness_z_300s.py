"""Explore-lane prototype spec (iter-003, depth-book lens).

depth_thickness_z_300s: trailing-300s z-score of log total 5-level
book size (bid + ask) -- the liquidity-SUPPLY state.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def compute(part: pl.DataFrame) -> pl.Series:
    b = pl.col("depth_bid5").cast(pl.Float64)
    a = pl.col("depth_ask5").cast(pl.Float64)
    tot = b + a
    x = (
        pl.when(tot > 0.0)
        .then(tot.log())
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
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
    name="depth_thickness_z_300s",
    mechanism=(
        "Liquidity-supply regime: total 5-level book size (bid + ask) is "
        "strongly non-stationary intraday (U-shaped liquidity), so only its "
        "deviation from its own trailing-300s distribution is informative. "
        "An unusually THICK book = competitive quoting / committed "
        "market-maker regime: flow is absorbed, impact is small, and recent "
        "moves tend to revert; an unusually THIN book = withdrawn liquidity "
        "where any aggression moves price (continuation of ongoing flow). "
        "Log transforms the heavy-tailed size; z-scoring makes it a regime "
        "state rather than a raw level (the dead-end class). Quantity-"
        "dimension companion of spread_z_300s (price dimension)."
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 family brief: liquidity-supply state leg; spread_z_300s "
        "built-in convention applied to book thickness; Bouchaud et al. "
        "(2004) impact inversely related to available liquidity; level-z "
        "separation per the iter-001/002 meta-lesson."
    ),
    compute=compute,
)
