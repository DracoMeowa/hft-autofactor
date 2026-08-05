"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_dev_extremeness: current absolute open-deviation relative to the
day's running maximum absolute deviation -- proximity to the day's most
one-sided state vs the open anchor.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-12  # below this peak the ratio is undefined (no stretch yet)


def compute(part: pl.DataFrame) -> pl.Series:
    """|dev_bps| / cum_max(|dev_bps|) in [0,1]; null until first stretch."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    adev = dev.abs()
    peak = adev.cum_max()
    out = (
        pl.when(peak.is_not_null() & (peak > EPS))
        .then(adev / peak)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="open_dev_extremeness",
    mechanism=(
        "Where the current stretch sits relative to the day's own record "
        "one-sidedness. Value near 1 = the deviation from the open is at "
        "or near its day-maximum: inventory is the most one-sided it has "
        "been all session, open-referenced benchmark desks are maximally "
        "stretched, and the market stands at the breakout-or-snap "
        "decision point where the anchored reversion force is strongest "
        "per bp of stretch -- historically the reversion IC of the open-"
        "deviation family should concentrate here. Value drifting toward "
        "0 = the current deviation is small FRACTION of the day's "
        "demonstrated stretch capacity: the anchor has already reasserted "
        "itself earlier in the session and price is operating well inside "
        "its envelope -- a calm, two-sided state. Dividing by the running "
        "peak makes quiet days and violent days comparable (a 10bp "
        "stretch is 'extreme' on a day whose max was 12bp and 'routine' "
        "on one whose max was 60bp), which neither the raw deviation "
        "level nor a rolling z provides: the reference is the whole-day "
        "envelope, matched to the whole-day anchor."
    ),
    info_set="mid_px, open_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 1 (deviation "
        "dynamics) in its envelope form; distance-to-day-extreme "
        "conditioning in the spirit of the round-1 finding that range-"
        "position states carry IC where width/recency variants die."
    ),
    compute=compute,
)
