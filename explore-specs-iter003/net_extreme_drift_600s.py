"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

net_extreme_drift_600s: signed net migration of the day's extremes over the
trailing 600s, in bps -- how many bps more (or less) the day-high moved UP
than the day-low moved DOWN. Size-weighted boundary pressure.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 200  # 200 x 3s rows = 600s differencing window


def compute(part: pl.DataFrame) -> pl.Series:
    """(diff_200(high_px) + diff_200(low_px)) / mid_px * 1e4; first 200 rows
    null. high diffs are >= 0 (running max), low diffs <= 0 (running min),
    so the sum is the net upside extremum migration."""
    d_hi = pl.col("high_px").diff(W)
    d_lo = pl.col("low_px").diff(W)
    val = (d_hi + d_lo) / pl.col("mid_px") * 1e4
    return part.select(val.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="net_extreme_drift_600s",
    mechanism=(
        "Signed net migration of the day's boundaries over ten minutes: how "
        "many bps more did the intraday high climb than the intraday low "
        "fell? High-side refreshes are trades lifting beyond every earlier "
        "price (buy-side stop runs, breakout attempts); low-side refreshes "
        "are sells pressing beneath every earlier price. The net imbalance, "
        "SIZE-weighted, reveals which side is consistently winning the "
        "boundary battle: the informed side leaves its footprint in "
        "repeated one-sided extreme migration, and stop-cascade dynamics "
        "make such migration persist over the following minutes. Magnitude "
        "weighting separates conviction (large extensions of the envelope) "
        "from single-tick probes, and the 600s span integrates a full "
        "order-flow episode rather than one burst."
    ),
    info_set="high_px, low_px, mid_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 4 (extreme-refresh "
        "functionals via cum/diff tricks) on the batch-2 feed extremes; "
        "size-weighted boundary pressure as the magnitude counterpart of "
        "the range-position champion family; 300-900s accumulation horizon "
        "per the round-1 meta-lesson."
    ),
    compute=compute,
)
