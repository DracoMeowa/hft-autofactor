"""Explore-lane prototype spec (iter-003 R6D, family R6D).

ofi_60s_zvel_extreme_15s: extremity-weighted z-VELOCITY product on the
standard 60s order-flow imbalance (ofi_60s). The 15s change rate of
z_300(ofi_60s) weighted by |z|. Applies the winning zvel-extreme template
to the primary OFI substrate.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG = 5


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 15s z-velocity of ofi_60s; warm-up null."""
    z = _z(pl.col("ofi_60s"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_60s_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted book-flow velocity: the 15s change rate of "
        "z_300(ofi_60s), weighted by how extreme the flow regime is "
        "(|z|). ofi_60s is the engine's standard order-book-delta flow "
        "over the trailing minute -- the workhorse OFI measure. When the "
        "minute-flow regime is already stretched (high |z|: one-sided "
        "book pressure running far beyond the 300s norm) and its z is "
        "moving fast (large dz), the sustained book-building program is "
        "being rapidly escalated -- informed passive flow that persisted "
        "for a minute and is now intensifying in seconds. This is a "
        "stronger pressure signal than either the level alone (dead per "
        "round-1: slow OFI levels carry no short-horizon IC) or the raw "
        "velocity (unweighted): the |z| weight ensures only extreme "
        "regimes contribute, re-ranking velocity by the crowdedness of "
        "the flow state it moves. Economically distinct from the admitted "
        "ofi_z_cross_vel_15s (event-sparse sign-flip velocity): this is "
        "always-on extremity-weighted velocity, firing on any fast motion "
        "of a stretched regime, not just at crossings. Also distinct from "
        "ofi_15s_zvel_extreme_15s (faster substrate): the 60s base "
        "captures minute-scale flow escalation, not quarter-minute bursts."
    ),
    info_set="ofi_60s",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel velocity substrate. "
        "ofi_60s is the primary OFI column. The zvel-extreme template "
        "produced the round-4 strongest short-horizon signals on wdi/oir; "
        "ofi_60s has only been used in level-z (ofi_15s_z_120s uses "
        "ofi_15s) and crossing-vel (ofi_z_cross_vel_15s) form, never in "
        "the always-on extremity-weighted velocity product."
    ),
    compute=compute,
)
