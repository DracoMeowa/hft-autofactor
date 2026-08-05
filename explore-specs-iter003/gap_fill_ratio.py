"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

gap_fill_ratio: sign-preserving log-compressed progress of mid through the
overnight gap -- (mid - pre_close)/(open - pre_close), 1 = gap untouched,
0 = exactly filled, <0 = overfilled past the pre-close.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS_GAP = 1e-3  # below ~1 tick of overnight gap the ratio is undefined


def compute(part: pl.DataFrame) -> pl.Series:
    """sign(r) * ln(1 + |r|) of the gap-progress ratio; null on ~zero
    gaps. The log compression keeps extreme small-gap ratios finite while
    preserving order and variation everywhere."""
    gap = pl.col("open_px") - pl.col("pre_close_px")
    ratio = (pl.col("mid_px") - pl.col("pre_close_px")) / gap
    val = ratio.sign() * (ratio.abs() + 1.0).log()
    out = (
        pl.when(gap.abs() >= EPS_GAP)
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="gap_fill_ratio",
    mechanism=(
        "The pre-close is the second natural day-anchor: the reference for "
        "overnight positions, ETF creation/redemption accounting and the "
        "previous close that index products settle against. The ratio "
        "(mid - pre_close)/(open - pre_close) measures how much of the "
        "overnight repricing the intraday session has GIVEN BACK: 1 = the "
        "opening gap intact, 0 = exactly filled, negative = overfilled. "
        "Gap-fill is a magnet with its own microstructure: fills attract "
        "counter-trend arbitrage and stop-trigger flows, and once the gap "
        "is exactly closed the original reason for the open price is "
        "consumed -- continuation beyond the fill (overfill) signals the "
        "overnight move is being REVERSED, not just filled, while a ratio "
        "persisting near 1 marks an overnight trend the intraday session "
        "has not yet contested. Scaling by the gap size makes small-gap "
        "and large-gap days comparable; the sign-preserving log keeps the "
        "ratio's ordering while bounding extreme values on tiny-gap days. "
        "Information carried by neither the open-deviation (anchor = "
        "open) nor bare momentum."
    ),
    info_set="mid_px, open_px, pre_close_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 2 (pre-close as "
        "second anchor, gap-fill pressure); intraday gap-closing "
        "regularities as in Lou, Polk & Skouras (2019) overnight/"
        "intraday return interplay, using the batch-2 pre_close_px "
        "pass-through."
    ),
    compute=compute,
)
