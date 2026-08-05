"""Explore-lane prototype spec (iter-003, etf-regime lens).

large_share_z_x_ti: z(large_trade_share_60s, 100 rows) x sign(trade
imbalance).  Informed-size flow WITH direction -- the dead iter-002
large_trade_share_level was the raw direction-free level; this is the
regime-relative surprise signed by aggressor flow.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 300s (100 x 3s rows) z window for the large-trade share
Z_WINDOW = 100


def compute(part: pl.DataFrame) -> pl.Series:
    """z(large-trade share) x sign(trade imbalance); warm-up rows null."""
    s = pl.col("large_trade_share_60s")
    mean = s.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = s.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    z = (s - mean) / std
    z = pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(z)
    ti = pl.col("trade_imbalance_60s")
    val = z * ti.sign()
    return part.select(
        pl.when(z.is_not_null() & ti.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="large_share_z_x_ti",
    mechanism=(
        "Informed-size flow is directional: institutional and informed "
        "traders cannot wait passively when information decays, so they "
        "lift/hit -- their large prints come with aggressor direction. An "
        "unusually high large-trade share RELATIVE to the trailing 300s "
        "size regime, signed by the concurrent aggressive-flow direction, "
        "flags informed participation on a specific side -> continuation "
        "of that side at 30-300s. Two deliberate departures from the dead "
        "iter-002 large_trade_share_level (raw level, direction-free): "
        "(1) z-scoring removes the persistent size-regime level that "
        "carried no IC (meta-lesson: levels of slow state are dead on "
        "588000); (2) multiplying by sign(trade_imbalance_60s) supplies "
        "the direction the level never had -- a size surge WITHOUT "
        "aggressor one-sidedness (two-sided repositioning) scores ~0."
    ),
    info_set="large_trade_share_60s, trade_imbalance_60s",
    inspiration=(
        "iter-002 archive: large_trade_share_level IC ~ 0 (direction-free "
        "level); iter-003 etf-regime brief: informed-size flow with "
        "direction. Trade-size decomposition of price impact (large prints "
        "concentrate information); large_trade_share_60s materialized "
        "2026-08-05 (wishlist)."
    ),
    compute=compute,
)
