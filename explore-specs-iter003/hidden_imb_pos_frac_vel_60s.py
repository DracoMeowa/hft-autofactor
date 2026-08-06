"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_imb_pos_frac_vel_60s: 60s momentum of the admitted
hidden_imb_pos_frac_300s occupancy -- is the hidden-bid-skew regime
CONSOLIDATING or ERODING (duration dynamics of the hidden skew).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing occupancy window (matches parent)
LAG = 20  # 20 x 3s rows = 60s momentum lag


def _hidden_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    den = hb + ha
    # Zero hidden depth on both sides -> neutral 0.0 (matches the admitted
    # parent's convention so the occupancy window stays complete).
    return pl.when(den > 0.0).then((hb - ha) / den).otherwise(pl.lit(0.0))


def compute(part: pl.DataFrame) -> pl.Series:
    """diff_60s of trailing-300s fraction of hidden imb > 0; warm-up null."""
    himb = _hidden_imb()
    pos = (
        pl.when(himb.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(himb > 0.0)
        .then(1.0)
        .otherwise(0.0)
    )
    frac = pos.rolling_mean(window_size=W, min_samples=W)
    return part.select(frac.diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_pos_frac_vel_60s",
    mechanism=(
        "Consolidation vs erosion of the hidden-bid-skew regime: the 60s "
        "change of the admitted hidden_imb_pos_frac_300s occupancy "
        "(trailing fraction of snapshots with the hidden layer net bid-"
        "skewed). The occupancy LEVEL says how consistently patient depth "
        "has leaned bid; its momentum says whether that lean is actively "
        "STRENGTHENING (accumulation posture deepening minute by minute -- "
        "demand conviction still building, supporting continued drift) or "
        "actively FADING (the entrenched hidden skew losing persistence -- "
        "patient demand quietly lifting, the slow-horizon support it "
        "provides starting to decay). Changing duration is second order: "
        "near-orthogonal to the occupancy level, to the hidden-side z "
        "levels, and to every momentum of raw quantities. This is momentum "
        "of an OCCUPANCY (the live facet), not momentum of raw hidden "
        "imbalance (hidden_imb_mom_60s/300s died IS-dead)."
    ),
    info_set="total_bid_vol, total_ask_vol, depth_bid5, depth_ask5 (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (e) applied to the hidden-skew "
        "occupancy; hidden_imb_pos_frac_300s admitted in round 3 -- this "
        "differentiates it (consolidation dynamics), avoiding the dead "
        "raw-momentum construction."
    ),
    compute=compute,
)
