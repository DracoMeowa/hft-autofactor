"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ofi_accel_z_180s: regime-relative book-flow acceleration -- z-score of the
30s-vs-60s OFI gap over the trailing 180s. Acceleration SURPRISE rather
than acceleration level.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s trailing z window on the acceleration gap


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_30s - ofi_60s, 180s); constant windows map to 0.0 (neutral)."""
    gap = pl.col("ofi_30s") - pl.col("ofi_60s")
    mean = gap.rolling_mean(window_size=W, min_samples=W)
    std = gap.rolling_std(window_size=W, min_samples=W)
    z = (gap - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_accel_z_180s",
    mechanism=(
        "Acceleration surprise: the 30s-minus-60s OFI gap measures how much "
        "faster the recent half-minute of book building runs versus the "
        "minute average; z-scoring that gap over the trailing 180s asks "
        "whether the CURRENT acceleration is unusual for this instrument-"
        "day. Moderate acceleration happens constantly inside trending "
        "tape and gets priced within seconds; the z-transform isolates the "
        "rare step-changes -- a fresh informed participant suddenly "
        "building the book faster than anything in the last three minutes "
        "-- and it is those episodes that continue at 15-60s. Symmetric "
        "negative surprises flag abrupt passive withdrawal / exhaustion. "
        "Distinct economic question from the raw gap (ofi_accel_15_60): "
        "raw measures the acceleration level in qty units, this measures "
        "acceleration relative to its own recent regime, so it fires only "
        "on changes of pace, not on sustained fast tape."
    ),
    info_set="ofi_30s, ofi_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R2-B brief direction 1 ('and their z' variants); "
        "regime-relative acceleration per the round-1 meta-lesson that "
        "conditions/deltas survive where levels die; MACD-surprise logic "
        "(Cartea-Jaimungal-Penalva 2015) with the genuine short-window "
        "engine columns instead of smoothed legs."
    ),
    compute=compute,
)
