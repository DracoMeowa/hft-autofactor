"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

netbuy_pressure_z_120s: regime-relative side-attributed net buying
pressure from the new buy_vol_60s / sell_vol_60s columns.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(netbuy_pressure, 120s); constant windows map to 0.0 (neutral)."""
    buy = pl.col("buy_vol_60s")
    sell = pl.col("sell_vol_60s")
    tot = buy + sell
    netbuy = (
        pl.when(tot.is_not_null() & (tot > 0.0))
        .then((buy - sell) / tot)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    mean = netbuy.rolling_mean(window_size=W, min_samples=W)
    std = netbuy.rolling_std(window_size=W, min_samples=W)
    z = (netbuy - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="netbuy_pressure_z_120s",
    mechanism=(
        "Unusual aggressive one-sidedness, regime-relative: the net buy "
        "pressure ratio built from the new side-attributed volumes is "
        "bounded [-1,1] and volume-scale-free; z-scoring it over the "
        "trailing 120s flags minutes where aggressive flow is unusually "
        "one-sided for THIS instrument-day rather than merely positive. "
        "Sustained buy pressure above the day's own norm means demand is "
        "repeatedly lifting offers faster than the recent regime -- the "
        "signature of continuation at 15-60s; unusually negative pressure "
        "flags distribution. The attribution of buy_vol/sell_vol differs "
        "from the engine trade_imbalance_60s normalization, so this is a "
        "re-derivation of aggressive direction from an independent "
        "decomposition; agreement between the two is itself informative, "
        "and the z-reference (not the level) keeps the factor regime-"
        "relative per the change-over-level lesson."
    ),
    info_set="buy_vol_60s, sell_vol_60s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 3 (net buy/sell pressure and its "
        "z); buy_vol_60s/sell_vol_60s materialized 2026-08-06; regime-"
        "relative state per round-1 conditions-over-levels lesson."
    ),
    compute=compute,
)
