"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

hidden_bid_support_z_300s: trailing-300s z-score of the BID-side hidden
share -- the fraction of total bid depth parked BEYOND the executable top-5
levels. Hidden bid support regime.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(hidden_bid / total_bid, 300s); warm-up rows null."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    share = (
        pl.when(tb > 0.0)
        .then(hb / tb)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(_z(share, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_bid_support_z_300s",
    mechanism=(
        "Hidden bid support regime: the fraction of the ENTIRE bid side that "
        "rests beyond the executable top-5 levels, z-scored against its own "
        "trailing-300s distribution. Side decomposition of the book that the "
        "imbalance lens cannot see: a bid side whose size is parked deep "
        "(high hidden share) is patient demand STACKED BELOW the touch -- "
        "buyers willing to queue far from execution rather than chase, which "
        "forms a durable reservoir that keeps refilling the visible bid and "
        "supports continued upward drift / a firm floor. A bid side "
        "concentrated at the executable tip with little behind it is exposed "
        "demand, fragile to consumption. This is a one-sided LEVEL of hidden "
        "placement, not a bid-vs-ask asymmetry (conc_imb_z) nor a full-book "
        "imbalance, so it is a distinct economic quantity; z-scoring keeps it "
        "in the live regime class rather than the dead raw-level class."
    ),
    info_set="depth_bid5, total_bid_vol (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 2 (side decomposition: hidden bid "
        "support); displayed vs undisplayed reserves (Buti & Rindi 2013); "
        "depth beyond level-5 confirmed as real incremental info in round 2, "
        "but only in ratio/z form."
    ),
    compute=compute,
)
