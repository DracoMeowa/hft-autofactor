"""Explore-lane prototype spec (iter-003, price-vol family).

price_accel_60_180: momentum acceleration -- 60s mid momentum minus 180s
mid momentum. Positive = the recent move is speeding up; negative = it is
running out of steam.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: rows for the two momentum windows (3s snapshots): 20 -> 60s, 60 -> 180s
K_FAST = 20
K_SLOW = 60


def compute(part: pl.DataFrame) -> pl.Series:
    """(mom_60s - mom_180s) of log-mid; warm-up rows null (diff semantics)."""
    log_mid = pl.col("mid_px").log()
    accel = log_mid.diff(K_FAST) - log_mid.diff(K_SLOW)
    return part.select(accel.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="price_accel_60_180",
    mechanism=(
        "Momentum acceleration / deceleration: the difference between the "
        "trailing-60s and trailing-180s log-mid returns measures how the "
        "velocity of the current move is changing. A positive value means "
        "the last minute moved faster than the last three minutes -- the "
        "move is accelerating, the signature of fresh directional "
        "information still being incorporated (continuation likely). A "
        "negative value means the move is decelerating -- exhaustion, "
        "liquidity re-forming, reversal risk. Level momentum confounds "
        "direction with persistence; the first difference of momentum "
        "isolates persistence itself, which is what predicts the next "
        "15-300s increment."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 price-vol family brief seed idea 4 (20-row minus 60-row "
        "mid log-momentum). Momentum-acceleration analogue of the registered "
        "ti_ewm_accel_120s (which accelerates trade imbalance, not price); "
        "price second-momentum / move-exhaustion logic."
    ),
    compute=compute,
)
