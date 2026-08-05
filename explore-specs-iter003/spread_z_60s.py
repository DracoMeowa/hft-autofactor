"""Explore-lane prototype spec (iter-003, price-vol family).

spread_z_60s: fast causal z-score of the quoted spread over 20 rows (60s).
Faster liquidity-state shift than the built-in spread_z_300s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 20 rows x 3s = 60s trailing window
W = 20


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
    name="spread_z_60s",
    mechanism=(
        "Fast liquidity-state shift: a quoted spread that is suddenly wide "
        "vs its own trailing-60s distribution flags an immediate quoting "
        "stress -- market makers pulling liquidity ahead of expected "
        "adverse selection. The built-in spread_z_300s averages over five "
        "minutes and is slow to react; a 60s window catches the widening "
        "within seconds of it starting. Spread widening precedes volatility "
        "bursts and often precedes the directional move that the wider "
        "quotes are protecting against, so the fast z should lead the slow "
        "one into 15-60s returns."
    ),
    info_set="quoted_spread_ticks (library)",
    inspiration=(
        "iter-003 price-vol family brief: 'spread_z_300s is TAKEN built-in; "
        "build the fast 60s/120s analogues'. Seed idea 11. Spread "
        "persistence and state-dependence (Stoll 2003); fast liquidity-"
        "state analogue of spread_z_300s."
    ),
    compute=compute,
)
