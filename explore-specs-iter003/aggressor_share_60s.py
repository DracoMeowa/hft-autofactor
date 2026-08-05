"""Explore-lane prototype spec (iter-003, price-vol family).

aggressor_share_60s: trailing-60s (20-row) net aggressor-side frequency --
share of snapshots with last above mid minus share with last below mid.
Sign-only (robust to trade-size outliers), unlike the magnitude-weighted
last_mid_gap_ma_30s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 20 rows x 3s = 60s trailing window
W = 20


def compute(part: pl.DataFrame) -> pl.Series:
    """rolling_mean_20( sign(last - mid) ); warm-up rows null."""
    sign_gap = (pl.col("last_px") - pl.col("mid_px")).sign()
    return part.select(
        sign_gap.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="aggressor_share_60s",
    mechanism=(
        "Aggressor consistency over the last minute: the net frequency of "
        "buyer-initiated vs seller-initiated snapshots (sign of last-mid), "
        "robust to a single oversized print because it counts direction, "
        "not size. A high positive value means buyers have been the "
        "aggressor on MOST recent snapshots -- consistent, repeated demand "
        "that works through the book -- which is the footprint of informed "
        "accumulation and predicts continuation. A value near zero means "
        "balanced two-sided flow (no edge). This frequency view complements "
        "the magnitude-weighted gap mean: consistency of aggression is "
        "distinct from intensity of aggression."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 price-vol family brief: aggressor side from last vs mid. "
        "Seed idea 10 (rolling 20-row share above/below mid). Sign-based "
        "aggressor balance -- order-flow sign persistence (Cont, Kukanov & "
        "Stoikov 2014)."
    ),
    compute=compute,
)
