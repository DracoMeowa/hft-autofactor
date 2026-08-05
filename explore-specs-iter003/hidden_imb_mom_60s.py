"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

hidden_imb_mom_60s: 60s momentum of the hidden-layer imbalance --
(full-book minus top-5 volume on each side), the willingness to queue
OUTSIDE the executable levels.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s momentum window


def _hidden_imb() -> pl.Expr:
    """(hidden_bid - hidden_ask) / (hidden_bid + hidden_ask), clipped >= 0."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    hb = pl.when(tb - db > 0.0).then(tb - db).otherwise(pl.lit(0.0))
    ha = pl.when(ta - da > 0.0).then(ta - da).otherwise(pl.lit(0.0))
    den = hb + ha
    return (
        pl.when(den > 0.0)
        .then((hb - ha) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """60s delta of hidden-layer imbalance; warm-up rows null."""
    return part.select(_hidden_imb().diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_mom_60s",
    mechanism=(
        "Hidden-depth pressure momentum: subtracting the 5-level totals "
        "from the full-book totals isolates the resting volume BEYOND the "
        "executable top of book -- the hidden layer. The 60s change of "
        "this layer's bid/ask imbalance measures how fast patient order "
        "flow is re-positioning outside the visible queue. Outer levels "
        "are posted deliberately and are not forced to execute, so a fresh "
        "one-sided build-up there is intent-bearing: net hidden bid "
        "pressure arriving now means patient buyers are stacking below the "
        "touch ahead of expected upward migration, and the touch follows "
        "the reservoir within 15-60s. Economically distinct from wdi/"
        "full-book momenta, which are dominated by the top levels, and "
        "from iceberg layering behind the touch (depth_ratio_5to1_z), "
        "which is symmetric and level-based."
    ),
    info_set="total_bid_vol, total_ask_vol, depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 R2-C family brief direction 2 (hidden depth beyond the "
        "5 levels and its imbalance diff); displayed vs undisplayed "
        "reserve literature (Buti & Rindi 2013); outer queue as intent "
        "signal; batch-2 total_*_vol materialized 2026-08-06."
    ),
    compute=compute,
)
