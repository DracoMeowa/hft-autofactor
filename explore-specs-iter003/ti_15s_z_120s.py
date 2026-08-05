"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ti_15s_z_120s: rolling z-score of the new 15s trade imbalance --
short-window AGGRESSION SURPRISE, resolved four times faster than the
panel's trade_imbalance_60s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(trade_imbalance_15s, 120s); constant windows map to 0.0 (neutral)."""
    x = pl.col("trade_imbalance_15s")
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
    name="ti_15s_z_120s",
    mechanism=(
        "Short-window aggression surprise: trade_imbalance_15s resolves the "
        "signed aggressive volume balance over quarter-minute windows, the "
        "timescale on which a single marketable program order dominates "
        "the tape. Z-scoring against the trailing 120s flags imbalance "
        "readings extreme relative to the instrument's own recent "
        "aggression regime. Extreme short-window imbalance is the "
        "footprint of an informed aggressor arriving NOW; Kyle (1985) "
        "persistence says the information keeps diffusing while the order "
        "works, so 15-60s drift continues in the imbalance direction. "
        "Distinct from the registered ti_ewm_state/accum cluster: those "
        "accumulate 60s imbalance into slow 120-300s state; this is the "
        "raw impulse, z-referenced locally, with no EWMA state anywhere. "
        "Redundancy note: the dedup gate will test rank correlation vs "
        "panel trade_imbalance_60s; the 1/4 window plus local z-reference "
        "are the decorrelation levers."
    ),
    info_set="trade_imbalance_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 2 (short-window rolling z); "
        "trade_imbalance_15s materialized 2026-08-06; Kyle (1985) informed-"
        "flow persistence at the fastest resolved scale."
    ),
    compute=compute,
)
