"""Explore-lane prototype spec (iter-003 R2-C, trade-structure lens).

trade_size_z_300s: trailing-300s z-score of log average trade size --
the granularity regime (institutional tickets vs retail dust).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(log avg_trade_size_60s, 300s); warm-up rows null."""
    size = pl.col("avg_trade_size_60s")
    x = (
        pl.when(size > 0.0)
        .then(size.log())
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(_z(x, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="trade_size_z_300s",
    mechanism=(
        "Granularity regime: the average trade size, log-transformed and "
        "z-scored against its own trailing-300s distribution, measures "
        "whether the tape is currently trading in institutional tickets "
        "or retail dust RELATIVE to its recent self. Unusually large "
        "average tickets on this ETF flag block/creation-redemption "
        "participation windows, which are largely inventory-driven and "
        "price-insensitive: market makers absorb such flow against their "
        "books, so the price pressure around size spikes is expected to "
        "partially REVERT at 60-900s (negative IC hypothesized). The "
        "z-vs-own-regime form removes the strong intraday non-stationarity "
        "of ticket size and stays in the live regime-state class -- the "
        "raw direction-free size LEVEL is exactly the class that died in "
        "iter-002 (large_trade_share_level)."
    ),
    info_set="avg_trade_size_60s (wishlist batch 1)",
    inspiration=(
        "iter-003 R2-C family brief direction 5 (trade granularity: "
        "avg_trade_size z); institutional-vs-retail granularity regime; "
        "iter-002 archive lesson: level statistics of size dead, regime-z "
        "is the live reformulation; log transform for heavy-tailed sizes."
    ),
    compute=compute,
)
