"""Explore-lane prototype spec (iter-003 R5-D, spread raw-level wide gate).

oir_mom_spread_wide_gate: 60s momentum of the top-of-book imbalance ratio
(oir) multiplied by the RAW spread level -- touch-queue turning under
absolute quoting cost.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s delta (matches oir_mom_60s library)


def compute(part: pl.DataFrame) -> pl.Series:
    """oir.diff(20) x raw quoted_spread_ticks; warm-up null."""
    base = pl.col("oir").diff(D)
    sp = pl.col("quoted_spread_ticks").cast(pl.Float64)
    return part.select((base * sp).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_mom_spread_wide_gate",
    mechanism=(
        "Touch-queue rotation amplified by absolute quoting cost: the "
        "60s change of the top-of-book imbalance ratio (oir) is "
        "multiplied by the RAW quoted_spread_ticks level. When the touch "
        "queue turns (bid-side weight gaining or losing relative to the "
        "ask-side) while the spread is WIDE in absolute terms, market "
        "makers are repositioning under explicit cost -- they are "
        "widening because of perceived risk AND rebuilding the queue "
        "in a new direction, a doubly-informed posture. Under tight "
        "1-tick quoting the same momentum is cheap routine rotation "
        "with lower information content. The raw-spread weight directly "
        "scales by the absolute cost makers are charging, which the "
        "z-gate normalizes away. oir_mom_60s was round-1 admitted (top-"
        "of-book momentum cluster) with no spread interaction yet. The "
        "raw-level product is near-orthogonal to the base (round-4 "
        "finding: panel rho 0.02-0.10 for raw-spread gates)."
    ),
    info_set="oir, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-D brief direction (a): wide_gate fill-in via RAW "
        "spread level. oir_mom_60s (round-1 admitted, top-of-book "
        "momentum) has no spread interaction yet."
    ),
    compute=compute,
)
