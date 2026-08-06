"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

gap_accel_60_180: ACCELERATION of the aggressor gap -- fast (60s) minus
slow (180s) gap change; the walk-through pace itself speeding up/down.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment (588000)
D_FAST = 20   # 20 x 3s rows = 60s
D_SLOW = 60   # 60 x 3s rows = 180s


def compute(part: pl.DataFrame) -> pl.Series:
    """diff(gap, 20) - diff(gap, 60); warm-up rows null (max D_SLOW)."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    accel = gap_ticks.diff(D_FAST) - gap_ticks.diff(D_SLOW)
    return part.select(accel.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="gap_accel_60_180",
    mechanism=(
        "Second-order aggressor dynamics: the first derivative of the "
        "last-mid gap says whether the aggressor edge is widening or "
        "shrinking; the SECOND derivative -- the last 60s of gap change "
        "minus the last 180s -- says whether that walk-through is itself "
        "speeding up or exhausting. Acceleration (fast change above slow "
        "change) marks escalation: participation arriving faster than the "
        "book can replenish, aggression compounding, typically the phase "
        "just before a queue breaks and price jumps -- continuation with "
        "urgency at 15-60s. Deceleration marks the absorption phase: the "
        "same direction of walk but with fading urgency, passive interest "
        "rebuilding faster than it is consumed -- the move is on borrowed "
        "time. Levels and first velocities cannot tell escalation from "
        "exhaustion when both have the same sign of velocity; the "
        "fast-minus-slow construction is the standard acceleration "
        "decomposition (as in the admitted price_accel_60_180) applied to "
        "the trade-vs-quote aggressor channel instead of the mid itself."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R4-D brief direction (c) gap acceleration; the admitted "
        "price_accel_60_180 (mid acceleration) motivates applying the "
        "fast-minus-slow second derivative to the aggressor gap instead; "
        "dedup note: derivative-of-derivative of the admitted raw level "
        "last_mid_gap_ticks -- distinct economic question (urgency, not "
        "position)."
    ),
    compute=compute,
)
