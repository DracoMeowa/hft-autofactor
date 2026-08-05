"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

range_width_z_300s: trailing-300s z-score of the day-range ratio
(high_px - low_px) / mid_px. Self-normalized expansion-state trigger.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_Z = 100  # 100 x 3s rows = 300s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(width) over 300s; constant windows -> 0.0; first 100 rows null."""
    width = (pl.col("high_px") - pl.col("low_px")) / pl.col("mid_px")
    mean = width.rolling_mean(window_size=W_Z, min_samples=W_Z)
    std = width.rolling_std(window_size=W_Z, min_samples=W_Z)
    z = (width - mean) / std
    out = pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(z)
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_width_z_300s",
    mechanism=(
        "Range-width state, self-normalized: (high-low)/mid measures how far "
        "the day's price discovery has stretched in range terms, and its "
        "trailing-300s z-score fires when the envelope is unusually wide or "
        "narrow relative to THIS day's own recent regime. Wide-for-recent "
        "means fresh information is actively expanding the trading envelope "
        "(discovery regime, in which volatility and drift cluster and tend "
        "to persist); narrow-for-recent means compression inside "
        "established bounds (consolidation, in which prior impulses decay "
        "and reversion dominates). The z-form strips the mechanical "
        "intraday growth of the range, turning a dead slow level into a "
        "regime trigger, and removes cross-day scale so only relative "
        "expansion state is ranked."
    ),
    info_set="high_px, low_px, mid_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief directions 1+7 (range-width "
        "state z/diff built on the batch-2 high/low pass-throughs); "
        "round-1 meta-lesson that deltas/relative states carry signal while "
        "slow levels die; volatility-clustering regime logic (Engle 1982)."
    ),
    compute=compute,
)
