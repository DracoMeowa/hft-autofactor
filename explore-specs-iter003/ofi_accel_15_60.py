"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ofi_accel_15_60: raw fast-minus-slow book-flow gap using the NEW batch-2
15s OFI column -- the freshest quarter-minute of order-book-delta flow minus
the trailing minute. Flow ACCELERATION in the passive channel.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """ofi_15s - ofi_60s; warm-up nulls propagate from the engine columns."""
    return part.select(
        (pl.col("ofi_15s") - pl.col("ofi_60s")).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_accel_15_60",
    mechanism=(
        "Book-flow acceleration: ofi_15s - ofi_60s contrasts the freshest "
        "15s of order-book-delta flow with the trailing minute. When the "
        "newest quarter-minute of book building is stronger than the "
        "minute average, passive demand is accelerating NOW; limit-flow "
        "impulses lead price (Cont-Kukanov-Stoikov 2014), so positive "
        "acceleration predicts 15-30s drift up as the queue reprices, and "
        "negative acceleration (fresh flow collapsing below the minute "
        "norm) flags exhaustion before the tape turns. Economically "
        "distinct from the rejected round-1 ofi_fast_slow, which smoothed "
        "the SAME 60s series with two rolling means: here the fast leg is "
        "the engine's true 15s tick window, reacting to new order events "
        "within one snapshot instead of averaging them away. Raw delta "
        "form follows the round-1 strongest cluster (fast book momenta "
        "passed as raw deltas, OOS IC 0.09-0.17)."
    ),
    info_set="ofi_15s, ofi_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R2-B brief direction 1 (fast-minus-slow flow rate); "
        "batch-2 column ofi_15s materialized 2026-08-06; refines the "
        "rejected ofi_fast_slow by replacing smoothed-60s legs with the "
        "genuine short-window engine column; CKS (2014) OFI leads price."
    ),
    compute=compute,
)
