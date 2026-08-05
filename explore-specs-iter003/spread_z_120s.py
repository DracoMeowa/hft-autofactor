"""Explore-lane prototype spec (iter-003, price-vol family).

spread_z_120s: mid-horizon causal z-score of the quoted spread over 40 rows
(120s). Between the fast spread_z_60s and the built-in spread_z_300s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 40 rows x 3s = 120s trailing window
W = 40


def compute(part: pl.DataFrame) -> pl.Series:
    x = pl.col("quoted_spread_ticks")
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
    name="spread_z_120s",
    mechanism=(
        "Mid-horizon liquidity-state shift: the trailing-120s z-score of "
        "the quoted spread sits between the fast 60s reaction and the "
        "built-in 300s baseline. It flags a widening that is sustained "
        "beyond a single burst (so not a one-snapshot artifact) but has not "
        "yet been absorbed into the five-minute baseline. Sustained spread "
        "widening over two minutes signals a persistent adverse-selection "
        "regime; such regimes accompany volatility clustering and precede "
        "the move the widened quotes anticipate, carrying into 30-300s "
        "returns."
    ),
    info_set="quoted_spread_ticks (library)",
    inspiration=(
        "iter-003 price-vol family brief: fast liquidity-state variants of "
        "the built-in spread_z_300s. Seed idea 12. Spread persistence "
        "(Stoll 2003); mid-window analogue bridging spread_z_60s and the "
        "built-in 300s."
    ),
    compute=compute,
)
