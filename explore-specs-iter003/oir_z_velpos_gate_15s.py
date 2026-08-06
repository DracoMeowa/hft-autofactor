"""Explore-lane prototype spec (iter-003 R5, family R5-B).

oir_z_velpos_gate_15s: NEW construction -- velocity-SIGN-gated z on oir.
The z-level of oir scored ONLY when its own 15s velocity is positive
(build-up regime), zeroed otherwise. Isolates the informed-positioning
half of the level distribution from the routine-churn half.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z when dz > 0, else 0.0; warm-up rows null.

    The build-up gate (dz > 0) selects rows where the z-regime is actively
    increasing. During decay or stagnation (dz <= 0), the output is
    exactly 0 -- no directional claim. Warm-up (z or dz null) is null,
    never zero-filled.
    """
    z = _z(pl.col("oir"), W)
    dz = z - z.shift(LAG)
    return part.select(
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(dz > 0.0)
        .then(z)
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="oir_z_velpos_gate_15s",
    mechanism=(
        "Build-up-isolated touch-imbalance level: z_300(oir) scored only "
        "when its own 15s z-velocity is positive (regime actively "
        "increasing), zeroed otherwise. The economic claim is strictly "
        "one-sided: during active build-up (dz > 0), the top-of-book "
        "imbalance level carries directional information because posting "
        "into a building queue tilt is an informed act -- the level "
        "continues in its direction at 15-60s. During decay or stagnation "
        "(dz <= 0), the same level is ambiguous noise and NO claim is "
        "made (output is exactly zero, not a reversal claim). This "
        "structurally isolates the informed-positioning episodes from the "
        "mass of routine quote churn, making the nonzero support a strict "
        "subset of the level distribution. Distinct from the dead bare "
        "oir level-z (round-1) and from library oir_zvel_extreme_15s "
        "(round-4 product form dz*|z|): the gate uses instantaneous "
        "velocity SIGN as a binary filter on the level, not as a "
        "magnitude weight."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R5-B family brief: velocity-sign-gated z construction; "
        "direction-isolated level via binary velocity gate on the "
        "admitted-round-4 winning base."
    ),
    compute=compute,
)
