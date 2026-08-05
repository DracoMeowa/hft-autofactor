"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

dayedge_x_ti60: DAY range-edge resolution gated by 60s aggressive flow
-- the day-anchor sibling of rollrpos_x_ti15 on a slower clock.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-12


def compute(part: pl.DataFrame) -> pl.Series:
    """|day_range_pos - 0.5| x 2 x trade_imbalance_60s; warm-up null."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    ext = (pos - 0.5).abs() * 2.0
    ti = pl.col("trade_imbalance_60s")
    return part.select((ext * ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="dayedge_x_ti60",
    mechanism=(
        "Day-edge resolution follows flow, on the slower aggression "
        "clock. The day's cumulative extremes are where stops, breakout "
        "orders and benchmark-triggered resting interest cluster. "
        "mid_day_range_pos passed all five horizons with NEGATIVE IC as "
        "the unconditional average -- but that average mixes edge rows "
        "with and without directional flow. Hypothesis: at the day "
        "edges, 60s aggressive imbalance decides the resolution -- "
        "buy imbalance at the day-high zone presses the breakout "
        "(continuation), sell imbalance cracks it (rejection); the "
        "mirror at the low. The signed factor |pos-0.5| x 2 x "
        "trade_imbalance_60s carries POSITIVE IC at 60-900s: conditioned "
        "on edge proximity, flow direction flips the average reversion "
        "sign. The amplitude gate keeps mid-range rows near zero (flow "
        "away from clusters is noise), so the factor re-ranks edge "
        "episodes by flow -- not a rescaling of the parent. Distinct "
        "from the dead range_pos_x_wdi (resting-book channel, linear "
        "product, IS-dead round 2): executed-aggression channel and "
        "extremeness gate."
    ),
    info_set="mid_px, trade_imbalance_60s",
    inspiration=(
        "iter-003 R3-D family brief direction 4 (range-extreme reversion "
        "with flow confirmation); round-1 all-horizon champion "
        "mid_day_range_pos; round-2 death map steered construction away "
        "from the dead linear day-pos x book-channel products."
    ),
    compute=compute,
)
