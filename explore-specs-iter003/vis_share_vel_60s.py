"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

vis_share_vel_60s: 60s VELOCITY of the visible share (top-5 depth / total
book depth) -- depth migrating between the executable touch and the hidden
layers (the level z of this share died; the migration direction is the
live question).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

LAG = 20  # 20 x 3s rows = 60s velocity lag


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
    """diff_60s of (depth_bid5+depth_ask5)/(total_bid+total_ask); warm-up null."""
    return part.select(_visible_share().diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="vis_share_vel_60s",
    mechanism=(
        "Liquidity migration between layers: the 60s change of the fraction "
        "of total book depth sitting in the executable top-5 levels. A "
        "RISING visible share is depth actively migrating from the hidden "
        "layers up to the touch -- participants competing to be executable, "
        "urgent interest surfacing where trades happen -- which precedes "
        "decisive short-horizon price action in the direction of the "
        "concurrent flow; a FALLING visible share is liquidity retreating "
        "into the deep/hidden layers -- a passive, absorbing posture where "
        "impact is blunted and moves stall. The LEVEL of this concentration "
        "ratio (visible_share_z_300s) died in round 3, consistent with the "
        "meta-lesson that level statistics of slow shape are dead while "
        "deltas live; the migration DIRECTION is the economic question the "
        "level discards. Sign-blind shape change, near-orthogonal to all "
        "imbalance/divergence factors by construction."
    ),
    info_set="depth_bid5, depth_ask5, total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (d): visible-share velocity; the "
        "dead visible_share_z_300s level motivates the derivative; the "
        "admitted div_x_vis_share shows the share carries interaction "
        "information but its own dynamics are unmined."
    ),
    compute=compute,
)
