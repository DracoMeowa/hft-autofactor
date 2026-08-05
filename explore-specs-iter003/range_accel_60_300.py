"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

range_accel_60_300: acceleration of intraday range expansion -- the 60s
change of the day-range ratio minus its 300s change. Second derivative of
the explored envelope: is information arrival speeding up or stalling?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_FAST = 20   # 20 x 3s rows = 60s
W_SLOW = 100  # 100 x 3s rows = 300s


def compute(part: pl.DataFrame) -> pl.Series:
    """diff_60s(width) - diff_300s(width); first 100 rows null."""
    width = (pl.col("high_px") - pl.col("low_px")) / pl.col("mid_px")
    fast = width.diff(W_FAST)
    slow = width.diff(W_SLOW)
    return part.select((fast - slow).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_accel_60_300",
    mechanism=(
        "Acceleration of range expansion: the 60s change of the day-range "
        "ratio minus its 300s change. Positive = discovery is speeding up "
        "(the envelope is growing faster now than its own recent pace), the "
        "signature of an emerging information or flow shock; volatility and "
        "information arrival cluster, so an accelerating burst tends to "
        "carry price further in the burst direction over the next minutes. "
        "Negative = expansion is decelerating: the shock is being absorbed, "
        "exhaustion sets in, and consolidation or reversal odds rise. The "
        "second-difference isolates the TURNING POINTS of the discovery "
        "regime, orthogonal to both the range level (dead slow state) and "
        "the first-difference speed."
    ),
    info_set="high_px, low_px, mid_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 1 (range expansion "
        "speed = information arrival); round-1 admitted price_accel_60_180 "
        "showed momentum ACCELERATION carries 900s signal, here the same "
        "second-derivative idea applied to the range envelope instead of "
        "the price path."
    ),
    compute=compute,
)
