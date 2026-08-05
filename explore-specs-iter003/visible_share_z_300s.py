"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

visible_share_z_300s: trailing-300s z-score of the top-5 share of TOTAL
book depth ((depth_bid5+depth_ask5)/(total_bid_vol+total_ask_vol)) -- the
book-concentration regime: how much of the whole book is packed at the
executable touch vs parked deep.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _visible_share() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((db + da) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(top-5 share of total book depth, 300s); warm-up rows null."""
    return part.select(_z(_visible_share(), W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="visible_share_z_300s",
    mechanism=(
        "Book-concentration regime: the fraction of the WHOLE book's depth "
        "that sits in the executable top-5 levels, z-scored against its "
        "trailing-300s distribution. When the top-5 share rises the book is "
        "CONCENTRATING at the touch (the deep queue thins relative to the "
        "visible tip); when it falls, depth is migrating outward. This is a "
        "scale-free, SIGN-BLIND shape variable: it measures where liquidity "
        "lives, not which side it favors, so it is orthogonal to every "
        "imbalance/divergence factor in the library. A touch-concentrated "
        "book means the executable layer is thick but there is little hidden "
        "buffer behind it -- once the visible tip is consumed, price walks "
        "far per unit flow; a deep-heavy book absorbs flow with reserves. "
        "The concentration level conditions impact magnitude and thus the "
        "persistence of ongoing pressure at 300-900s."
    ),
    info_set="depth_bid5, depth_ask5, total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 3 (top-5 share of total depth and its "
        "regime; depth thinning); book-shape distribution across levels "
        "(Zovko & Farmer 2002); distinct from depth_thickness_z_300s "
        "(absolute size) and depth_ratio_5to1_z (5-vs-1 layering)."
    ),
    compute=compute,
)
