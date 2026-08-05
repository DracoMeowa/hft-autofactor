"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

abs_open_dev_delta_300s: 300s change in the ABSOLUTE open-deviation -- the
speed at which mid is stretching AWAY from (+) or returning TOWARD (-) the
open, regardless of which side of the open the day is on.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s delta window


def compute(part: pl.DataFrame) -> pl.Series:
    """|dev_bps|(i) - |dev_bps|(i-100); warm-up rows null."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    return part.select(dev.abs().diff(W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="abs_open_dev_delta_300s",
    mechanism=(
        "Speed of the anchor tug-of-war, signed by direction of travel "
        "relative to the open rather than by price direction. Rising "
        "|deviation| = mid actively stretching away from the opening "
        "consensus on EITHER side (the anchor's pull is being overcome: "
        "fresh information or persistent inventory pressure is carrying "
        "price into unanchored territory, where the round-2 negative-IC "
        "reversion pull has NOT yet won); falling |deviation| = active "
        "reversion in progress (the anchored value is reasserting, and "
        "reversion episodes toward salient anchors tend to persist until "
        "the gap is closed). Plain momentum cannot separate these: a down "
        "move is reversion when mid is above the open but extension when "
        "below -- the absolute-value fold around the anchor is exactly "
        "that disambiguation, which is also why this is not a momentum "
        "clone (the transform is nonlinear in mid)."
    ),
    info_set="mid_px, open_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 1 (speed of stretch "
        "away from / back toward the open); anchor-reversion persistence "
        "logic as in intraday-VWAP/anchor price magnetism (Gao, Han, Li & "
        "Zhou 2018), expressed as a delta per the round-1 meta-lesson "
        "(deltas/states, not levels)."
    ),
    compute=compute,
)
