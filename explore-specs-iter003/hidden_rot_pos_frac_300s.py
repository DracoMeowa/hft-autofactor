"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_rot_pos_frac_300s: trailing-300s OCCUPANCY of bid-directed rotation
-- the share of time the 60s hidden side-gap velocity pointed toward the bid
side (duration of the rotation regime, orthogonal to its magnitude).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 300s trailing occupancy window
ZL = 20   # 60s side-gap velocity lag


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _hidden_bid_share() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    return (
        pl.when(tb > 0.0)
        .then(hb / tb)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _hidden_ask_share() -> pl.Expr:
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    return (
        pl.when(ta > 0.0)
        .then(ha / ta)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """Fraction of last-300s rows with 60s side-gap velocity > 0; warm-up null."""
    gap = _z(_hidden_bid_share(), W) - _z(_hidden_ask_share(), W)
    vel = gap.diff(ZL)
    pos = (
        pl.when(vel.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(vel > 0.0)
        .then(1.0)
        .otherwise(0.0)
    )
    frac = pos.rolling_mean(window_size=W, min_samples=W)
    return part.select(frac.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_rot_pos_frac_300s",
    mechanism=(
        "Duration of the hidden-rotation regime: the fraction of the "
        "trailing 300s during which the 60s velocity of the support-minus-"
        "supply z-gap was bid-directed. hidden_side_gap_vel_60s measures "
        "how fast hidden liquidity is rotating RIGHT NOW; this occupancy "
        "measures how CONSISTENTLY it has rotated toward the bid side over "
        "five minutes. A persistently bid-directed rotation is a structural "
        "rebalancing of patient capital -- demand repeatedly gaining hidden "
        "ground, not one transient stack event -- and such entrenched "
        "migration decays slowly, conditioning the next 300-900s toward "
        "continued upward drift / a firm floor. Occupancy is the "
        "magnitude-orthogonal decomposition that worked for div_pos_frac "
        "and hidden_imb_pos_frac in round 3: a weak-but-constant rotation "
        "and a violent-but-oscillating one read identically on the "
        "instantaneous velocity yet imply different commitment. Distinct "
        "from hidden_imb_pos_frac (occupancy of the imbalance LEVEL sign) "
        "because the base here is the sign of a VELOCITY of share-gap."
    ),
    info_set="total_bid_vol, total_ask_vol, depth_bid5, depth_ask5 (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (b) + round-3 winning pattern: "
        "occupancy/duration transforms of the hidden cluster (div_pos_frac, "
        "hidden_imb_pos_frac admitted) applied to the NEW rotation "
        "velocity; slow-regime horizons per eval-v2."
    ),
    compute=compute,
)
