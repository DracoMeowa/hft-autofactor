"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

rollrpos_x_ti15: LOCAL range-edge resolution gated by fast aggressive
flow -- at the rolling battle-range boundary, resolution follows the
flow direction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100     # 100 x 3s rows = 300s rolling range window
EPS = 1e-12


def compute(part: pl.DataFrame) -> pl.Series:
    """|roll_range_pos - 0.5| x 2 x trade_imbalance_15s; warm-up null."""
    mid = pl.col("mid_px")
    rmax = mid.rolling_max(window_size=W, min_samples=W)
    rmin = mid.rolling_min(window_size=W, min_samples=W)
    rng = rmax - rmin
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - rmin) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    ext = (pos - 0.5).abs() * 2.0
    ti = pl.col("trade_imbalance_15s")
    return part.select((ext * ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="rollrpos_x_ti15",
    mechanism=(
        "Local-edge resolution follows fast flow. mid_roll_range_pos_300s "
        "(admitted at 900s) marks the boundaries of the RECENT battle "
        "range, which intraday scalpers and execution algos defend and "
        "re-test many times a day. Hypothesis: at a local edge, the "
        "direction of contemporaneous aggressive execution DECIDES the "
        "resolution -- buy pressure at the local ceiling is breakout "
        "fuel, sell pressure a confident rejection; at the local floor "
        "buy pressure is defense/absorption, sell pressure breakdown. "
        "The signed factor |pos-0.5| x 2 x ti_15s carries POSITIVE IC: "
        "flow direction wins at the edges, against the bare "
        "range-position average-reversion tendency. The AMPLITUDE gate "
        "compresses mid-range rows (no boundary in play -> flow is "
        "noise) toward zero, localizing fast-flow information at the "
        "boundaries. Structurally distinct from the round-2-dead "
        "range_pos_x_ofi_z / range_pos_x_wdi (linear centered products "
        "on the DAY position with book channels, IS-dead): different "
        "anchor (rolling), different channel (executed aggression), "
        "amplitude gate instead of a sign-flipping product."
    ),
    info_set="mid_px, trade_imbalance_15s",
    inspiration=(
        "iter-003 R3-D family brief direction 4 (range position "
        "conditioned on flow direction with flow confirmation); round-2 "
        "admitted mid_roll_range_pos_300s; round-2 death map steered the "
        "construction away from the dead linear day-pos x flow products."
    ),
    compute=compute,
)
