"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

div_pos_frac_vel_60s: 60s momentum of the admitted div_pos_frac_300s
occupancy -- is the touch-leading-queue regime CONSOLIDATING or ERODING
(duration dynamics, second order to the level occupancy).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing occupancy window (matches parent)
LAG = 20  # 20 x 3s rows = 60s momentum lag


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """diff_60s of trailing-300s fraction of (wdi - fbi) > 0; warm-up null."""
    div = pl.col("wdi") - _fullbook_imb()
    pos = (
        pl.when(div.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(div > 0.0)
        .then(1.0)
        .otherwise(0.0)
    )
    frac = pos.rolling_mean(window_size=W, min_samples=W)
    return part.select(frac.diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="div_pos_frac_vel_60s",
    mechanism=(
        "Consolidation vs erosion of the touch-leading-queue regime: the "
        "60s change of the admitted div_pos_frac_300s occupancy (trailing "
        "fraction of snapshots with wdi above the full-book imbalance). "
        "The occupancy LEVEL says how entrenched the regime is; its "
        "momentum says whether entrenchment is actively INCREASING (the "
        "structural posture still forming -- displayed liquidity "
        "increasingly out front of the hidden queue, conditioning continued "
        "behavior of that regime) or actively UNWINDING (the entrenched "
        "posture started to erode within the last minute -- everything "
        "conditioned on it, including the slow-horizon trades it supports, "
        "is at risk of reversal). A duration statistic changing direction "
        "is a second-order quantity: near-orthogonal to the occupancy "
        "level itself and to divergence magnitude, and unlike the dead "
        "top5_book_div_mom_60s (momentum of the divergence MAGNITUDE) this "
        "is momentum of an occupancy -- the duration facet that round 3 "
        "proved live where magnitude momentum failed."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (e): div_pos_frac momentum; "
        "div_pos_frac_300s admitted in round 3 -- this differentiates the "
        "occupancy (consolidation dynamics), avoiding the dead magnitude-"
        "momentum construction."
    ),
    compute=compute,
)
