"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ti_accel_15_60: fast-minus-slow AGGRESSIVE-flow gap using the new batch-2
15s trade-imbalance column -- trade_imbalance_15s - trade_imbalance_60s.
Active-channel counterpart of ofi_accel_15_60.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """trade_imbalance_15s - trade_imbalance_60s; both bounded [-1,1]."""
    return part.select(
        (pl.col("trade_imbalance_15s") - pl.col("trade_imbalance_60s"))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ti_accel_15_60",
    mechanism=(
        "Aggressor-flow acceleration: trade_imbalance_15s - "
        "trade_imbalance_60s contrasts the freshest 15s of signed "
        "aggressive volume balance with the trailing minute. A positive "
        "value means marketable buying is intensifying RIGHT NOW relative "
        "to the minute context -- the aggressor is sweeping the touch "
        "faster than the recent pace, which consumes resting liquidity and "
        "precedes 15-30s continuation in the aggression direction. A "
        "negative value flags aggression exhaustion (buyers fading mid-"
        "minute) ahead of stalls or fades. Both legs are bounded in "
        "[-1,1], so the gap is scale-free across instruments and volume "
        "regimes (unlike qty-unit OFI gaps). Distinct channel from "
        "ofi_accel_15_60 by construction: accelerating aggression against "
        "decelerating book building is the urgent 'hit while liquidity "
        "pulls' state, and the two accelerations are only moderately "
        "correlated. Not an EWMA state (ti_ewm_accel_120s exists): this is "
        "a pure two-window engine-column difference at the fastest "
        "available resolution."
    ),
    info_set="trade_imbalance_15s, trade_imbalance_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R2-B brief direction 1 (fast-minus-slow flow rate); "
        "trade_imbalance_15s materialized 2026-08-06; the round-1 book-"
        "momentum cluster (oir/wdi/depth deltas, OOS IC 0.09-0.17) recast "
        "as trade-channel acceleration with the new 15s engine column."
    ),
    compute=compute,
)
