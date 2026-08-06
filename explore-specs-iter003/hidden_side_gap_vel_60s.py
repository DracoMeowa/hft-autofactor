"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_side_gap_vel_60s: 60s velocity of the support-minus-supply z-gap --
hidden liquidity actively ROTATING between the bid and ask sides.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing z window on each side's share
LAG = 20  # 20 x 3s rows = 60s velocity lag


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
    """diff_60s of [z(bid hidden share) - z(ask hidden share)]; warm-up null."""
    gap = _z(_hidden_bid_share(), W) - _z(_hidden_ask_share(), W)
    return part.select(gap.diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_side_gap_vel_60s",
    mechanism=(
        "Hidden liquidity rotating sides: the 60s change of the support-"
        "minus-supply gap, where each side is its own admitted hidden-share "
        "z (z of bid-side hidden share minus z of ask-side hidden share). "
        "Positive velocity means patient depth is actively rotating TOWARD "
        "the bid side -- hidden support building faster than hidden supply "
        "(or supply draining faster than support) -- a rebalancing of "
        "resting commitment in favor of demand that precedes continued "
        "upward pressure; negative velocity is rotation toward supply. The "
        "rotation is a FLOW-between-states quantity no level can capture: "
        "both sides can be simultaneously deep (two-sided reservoir) yet "
        "only one side is actively gaining. This is NOT the dead net hidden "
        "imbalance -- that was the LEVEL of (hb-ha)/(hb+ha) on absolute "
        "quantities (hidden_imb_z OOS-collapsed, its momenta died IS); here "
        "the inputs are side-normalized SHARES in z-space, and the operator "
        "is a velocity of their gap, a different economic question "
        "(direction of migration, not static skew)."
    ),
    info_set="total_bid_vol, total_ask_vol, depth_bid5, depth_ask5 (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (b): bid-vs-ask hidden balance "
        "SHIFTS (support minus supply, or its momentum); built on the two "
        "admitted one-sided R3-B z regimes; avoids the dead net-hidden-"
        "imbalance construction by differentiating the share-gap."
    ),
    compute=compute,
)
