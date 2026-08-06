"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

vis_share_vel_15s: FAST 15s velocity of the visible share -- sudden bursts
of touch-concentration (competitive repricing / queue-jump pulses).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

LAG = 5  # 5 x 3s rows = 15s velocity lag


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
    """diff_15s of (depth_bid5+depth_ask5)/(total_bid+total_ask); warm-up null."""
    return part.select(_visible_share().diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="vis_share_vel_15s",
    mechanism=(
        "Touch-concentration bursts: the 15s change of the fraction of "
        "total book depth parked in the executable top-5 levels. Fast "
        "upward jumps mean a sudden pulse of orders arriving AT the touch "
        "(competitive repricing, queue-jumping ahead of expected action) -- "
        "urgent executable interest that resolves within seconds; fast "
        "downward pulses mean the touch is being pulled/emptied into the "
        "hidden layers just as quickly -- imminent thinning of the "
        "executable book. The 15s clock isolates these discrete "
        "surfacing/retreat events from the slower drift of positioning, "
        "targeting the 15-30s horizons where fast book state paid off in "
        "round 1. Level of the concentration ratio is dead (round 3), its "
        "60s migration is the (d)-60s sibling; this fast variant is the "
        "event detector of the same migration process, a different "
        "economic clock rather than a rewindowed copy."
    ),
    info_set="depth_bid5, depth_ask5, total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (d): visible-share velocity at the "
        "fast clock; round-1 lesson that 15-30s horizons reward fast book "
        "state; dead visible_share_z_300s level motivates the derivative."
    ),
    compute=compute,
)
