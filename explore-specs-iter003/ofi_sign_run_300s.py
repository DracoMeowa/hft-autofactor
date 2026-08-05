"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ofi_sign_run_300s: net signed run of ofi_15s signs over the trailing 300s --
rolling SUM of sign(ofi_15s), magnitude-free directional commitment of the
book channel on the slow horizon.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing run window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing mean of sign(ofi_15s) over 100 rows == sum(signs)/100.

    Values in [-1, 1]: +1 = every 15s flow reading in the last five
    minutes was buy-side, -1 every one sell-side, 0 balanced/alternating.
    Warm-up rows null (min_samples=W), never zero-filled.
    """
    sgn = pl.col("ofi_15s").sign().cast(pl.Float64)
    return part.select(
        sgn.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_sign_run_300s",
    mechanism=(
        "Magnitude-free directional commitment of the book channel over "
        "five minutes: the rolling sum of sign(ofi_15s) counts how many of "
        "the last hundred quarter-minute flow windows pointed each way, "
        "ignoring how LARGE each was. This separates two regimes that "
        "magnitude accumulators conflate: a steady one-sided queue-"
        "building regime (run near +/-1) is the signature of patient "
        "limit-flow investment -- informed agents who prefer resting "
        "orders keep rebuilding one side of the book for minutes (Kyle "
        "1985 slicing via the passive channel) -- and such regimes "
        "continue drifting at 300-900s because the schedule behind them "
        "is unfinished; a balanced run (~0) is churn with no committed "
        "direction. It is not a raw accumulation (the dead ti15_accum/"
        "netvol_accum family summed MAGNITUDES and collapsed OOS): the "
        "sign run is bounded, regime-like, and a state rather than a "
        "sum of impulses. It differs from the dead ofi_pos_frac_300s "
        "(fraction of ofi_60s > 0): here the signs are read off the 4x "
        "faster ofi_15s column, so flips that the 60s window averages "
        "away still break the run -- finer-grained commitment timing."
    ),
    info_set="ofi_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R3-C brief direction 1 (signed run statistics via "
        "rolling sums of sign); Kyle (1985) patient informed slicing; "
        "round-2 lesson that slow horizons reward STATE statistics, not "
        "raw magnitude sums."
    ),
    compute=compute,
)
