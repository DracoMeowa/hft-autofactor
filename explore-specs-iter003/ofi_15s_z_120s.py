"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ofi_15s_z_120s: rolling z-score of the new 15s order-flow imbalance --
short-window BOOK-FLOW SURPRISE, resolved four times faster than the
panel's ofi_60s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_15s, 120s); constant windows map to 0.0 (neutral)."""
    x = pl.col("ofi_15s")
    mean = x.rolling_mean(window_size=W, min_samples=W)
    std = x.rolling_std(window_size=W, min_samples=W)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_15s_z_120s",
    mechanism=(
        "Short-window book-flow surprise: ofi_15s resolves order-book-"
        "delta flow over quarter-minute windows, where single placement/"
        "cancel bursts still dominate, instead of the minute-averaged "
        "ofi_60s that smooths them away. Z-scoring against the trailing "
        "120s converts qty-scale flow into a comparable pressure-surprise: "
        "a reading of +2 says the book is being built one way at a rate "
        "two standard deviations above this instrument's last two minutes. "
        "Informed limit flow arrives in exactly such bursts (CKS 2014: OFI "
        "drives short-horizon price changes) and precedes 15-60s "
        "continuation in the burst direction. The short z-window keeps the "
        "reference regime local so the signal tracks intraday regime "
        "changes instead of the whole-day norm. Redundancy control: ofi_15s "
        "shares its formula with panel ofi_60s but over 1/4 of the window, "
        "and the 120s z removes the shared slow regime component; if rank "
        "correlation with ofi_60s still clears 0.85 the dedup gate will "
        "catch it, but the burst resolution is genuinely new information."
    ),
    info_set="ofi_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 2 (short-window rolling z); "
        "ofi_15s materialized 2026-08-06; Cont-Kukanov-Stoikov (2014) OFI "
        "price impact at the fastest resolved scale."
    ),
    compute=compute,
)
