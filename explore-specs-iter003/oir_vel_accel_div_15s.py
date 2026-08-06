"""Explore-lane prototype spec (iter-003 R6, family R6B).

oir_vel_accel_div_15s: velocity-over-acceleration DOMINANCE ratio on the
top-of-book imbalance. z(15s z-velocity) divided by (1 + |z(15s z-accel)|):
large when the touch-rebuild velocity is extreme but its acceleration has
faded (momentum running ahead of its curvature fuel -- overextended), shrunk
when acceleration still backs the velocity (still fueled). Sign carried by
velocity.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity and acceleration


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(dz) / (1 + |z(d2z)|): velocity-dominance over faded acceleration.

    Denominator >= 1.0 so the ratio is bounded and never blows up; it is
    maximal when the velocity z is extreme while the acceleration z is near
    zero (curvature gone). Warm-up rows null.
    """
    z_e = _z(pl.col("oir"), W)
    dz_e = z_e - z_e.shift(LAG)
    d2z_e = dz_e - dz_e.shift(LAG)
    tmp = part.select(
        z_e.alias("_z"), dz_e.alias("_dz"), d2z_e.alias("_d2z")
    )
    tmp = tmp.select(
        _z(pl.col("_dz"), W).alias("_zdzz"),
        _z(pl.col("_d2z"), W).alias("_zd2zz"),
    )
    return tmp.select(
        (pl.col("_zdzz") / (1.0 + pl.col("_zd2zz").abs())).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="oir_vel_accel_div_15s",
    mechanism=(
        "Touch-rebuild overextension vs its curvature fuel: the z of the "
        "15s z-velocity of oir divided by (1 + |z of its 15s z-acceleration|). "
        "The best-quote tilt rebuild has a velocity (how fast the touch is "
        "moving) and an acceleration (whether that motion is speeding up). "
        "When the velocity regime reads extreme but the acceleration regime "
        "has faded toward zero (denominator ~1), the touch motion is "
        "running ahead of its own curvature fuel -- a velocity that nothing "
        "is still pushing, characteristic of an overextended thrust about to "
        "exhaust and revert; when the acceleration is still strong, the "
        "denominator grows and SHRINKS the ratio, correctly demoting the "
        "still-fueled moves. The sign is carried by the velocity, so the "
        "direction of the overextension is preserved. Distinct from the "
        "agreement product oir_zvel_x_zaccel_15s (multiplies the two z's, so "
        "strong acceleration AMPLIFIES) and from oir_zvel_div_15s (level "
        "minus velocity): here acceleration ENTERS THE DENOMINATOR, "
        "isolating velocity-dominant-over-faded-curvature -- the exhaustion "
        "question, the opposite of co-intensification."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6-B family brief: velocity/acceleration ratio "
        "(overextension when vel high but accel fading) on the oir base; "
        "division is the dual of the agreement product."
    ),
    compute=compute,
)
