"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ltns_z_180s: regime-relative SIGNED large-trade footprint from the new
large_trade_net_share_60s column (signed net share of the largest ~10%
of prints).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(large_trade_net_share_60s, 180s); constant windows -> 0.0."""
    x = pl.col("large_trade_net_share_60s")
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
    name="ltns_z_180s",
    mechanism=(
        "Signed institutional footprint surprise: large_trade_net_share_60s "
        "attributes a SIGNED net direction to the largest ~10% of prints "
        "(+ when the biggest tickets are buyer-initiated). Unlike the "
        "direction-free large_trade_share_60s whose LEVEL died in iter-002 "
        "(no sign = no directional information), this column carries the "
        "direction of whale flow. Z-scoring over 180s flags episodes where "
        "big-ticket flow is unusually one-sided versus the day's recent "
        "norm. Institutional execution shows up in the tail of the size "
        "distribution (information events are not split to dust), so an "
        "unusually signed tail is the cleanest informed-activity proxy "
        "available on this panel, and its direction should persist into "
        "15-60s returns while the program works."
    ),
    info_set="large_trade_net_share_60s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 4 (z of signed large-trade net "
        "share); large_trade_net_share_60s materialized 2026-08-06; "
        "signed repair of the dead direction-free large_trade_share_level; "
        "Bouchaud et al. (2009) anomalous impact of large trades."
    ),
    compute=compute,
)
