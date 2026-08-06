"""Explore-lane prototype spec (iter-003 R5, family R5-B).

oir_z_x_velfrac_15s: NEW construction -- level x rolling velocity-direction
persistence. z(oir) multiplied by (2 * pos_frac(dz, 300s) - 1), where
pos_frac is the trailing fraction of snapshots with positive z-velocity.
Tests whether sustained velocity direction (high persistence) isolates
informed build-up/decay better than instantaneous velocity sign.
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
    """z * (2 * pos_frac - 1) where pos_frac = rolling mean of (dz > 0).

    pos_frac is the fraction of the last W snapshots with positive z-
    velocity (1.0 during sustained build-up, 0.0 during sustained decay,
    0.5 during oscillation). Centered to [-1, 1] and multiplied by z.
    Warm-up rows null (z warm-up propagates through dz, then the rolling
    fraction requires W further non-null dz values).
    """
    z = _z(pl.col("oir"), W)
    dz = z - z.shift(LAG)
    indicator = (
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(dz > 0.0)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
    )
    pos_frac = indicator.rolling_mean(window_size=W, min_samples=W)
    centered = (pos_frac - pl.lit(0.5)) * pl.lit(2.0)
    return part.select((z * centered).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_z_x_velfrac_15s",
    mechanism=(
        "Level weighted by velocity-direction persistence: z_300(oir) "
        "multiplied by (2 * pos_frac - 1), where pos_frac is the trailing "
        "fraction of snapshots with positive 15s z-velocity. pos_frac = 1 "
        "means every recent snapshot saw the regime INCREASING (sustained "
        "build-up); pos_frac = 0 means sustained decay; pos_frac = 0.5 "
        "means oscillation. A sustained one-directional touch-imbalance "
        "regime (high persistence) with a stretched level marks committed "
        "institutional repositioning -- the level continues at 15-60s. A "
        "contested regime (pos_frac near 0.5) with the same level is "
        "two-sided churn with no directional edge, scored near zero by "
        "the centered persistence. Distinct from the binary gate "
        "oir_z_velpos_gate_15s (instantaneous dz sign): this uses the "
        "rolling FRACTION of positive-velocity snapshots, a softer "
        "persistence measure that distinguishes sustained from sporadic "
        "build-up. Also distinct from library ofi_sign_persist_60s (sign-"
        "run persistence of OFI): different base, different statistic "
        "(direction fraction vs consecutive-run fraction)."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R5-B family brief: velocity-direction-persistence x "
        "level construction; tests whether SUSTAINED velocity direction "
        "(windowed fraction) is more informative than instantaneous sign."
    ),
    compute=compute,
)
