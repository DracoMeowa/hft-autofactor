"""Explore-lane prototype spec (iter-003, price-vol family).

last_mid_gap_ma_30s: trailing-30s (10-row) mean of the raw aggressor gap
(last_px - mid_px)/tick.  Sustained aggressor direction, magnitude-weighted.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: SSE ETF minimum price increment
TICK = 0.001
#: 10 rows x 3s = 30s trailing window
W = 10


def compute(part: pl.DataFrame) -> pl.Series:
    """rolling_mean_10( (last-mid)/tick ); warm-up rows null."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    return part.select(
        gap_ticks.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="last_mid_gap_ma_30s",
    mechanism=(
        "Sustained aggressor direction: a single last-trade aggressor read "
        "is noisy (one snapshot can be an isolated fill), so the trailing-"
        "30s mean of the signed gap aggregates whether the recent prints "
        "have been consistently buyer- or seller-initiated. Persistent net "
        "buyer aggression over 30s means demand is repeatedly crossing to "
        "pay up -- informed buying that has not yet been absorbed -- and "
        "predicts short-horizon upward continuation; persistent seller "
        "aggression is the mirror. Magnitude-weighted (wide-quote "
        "aggression counts more), unlike the pure sign-frequency variant."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 price-vol family brief: aggressor side from last vs mid. "
        "Seed idea 9 (10-row mean of the raw gap). Smoothed aggressor "
        "pressure -- Lee & Ready (1991) direction smoothed over a fast "
        "window."
    ),
    compute=compute,
)
