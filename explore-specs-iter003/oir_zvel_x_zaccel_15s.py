"""Explore-lane prototype spec (iter-003 R6, family R6B).

oir_zvel_x_zaccel_15s: velocity-vs-acceleration AGREEMENT product on the
top-of-book imbalance. z(15s z-velocity) crossed with z(15s z-acceleration):
large positive when both derivatives point the same way at co-extreme
strength (momentum-with-curvature-fuel), negative when they oppose (turning
point). Continuous co-movement measure -- distinct from the gated
disagree-only twin and from the level-weighted velocity/accel products.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback for velocity and acceleration


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(dz) * z(d2z) where dz = 15s z-velocity, d2z = 15s z-acceleration.

    Both derivatives are z-normalized against their OWN 300s distributions,
    then crossed. Warm-up rows null (z warm-up propagates through two shifts
    and two more z windows).
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
    return tmp.select((pl.col("_zdzz") * pl.col("_zd2zz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zvel_x_zaccel_15s",
    mechanism=(
        "Velocity-acceleration co-intensification at the touch: the product "
        "of the trailing-300s z of the 15s z-velocity of oir with the z of "
        "its own 15s z-acceleration. The best-quote qty ratio is the most "
        "actively managed slot; its z-velocity is how fast the touch tilt is "
        "being rebuilt, its z-acceleration is whether that rebuild is "
        "speeding up. When BOTH derivatives read co-extreme in the SAME "
        "direction (product large positive), the repositioning carries its "
        "own curvature fuel -- fresh quoting is accelerating into the same "
        "direction with positive feedback -- and the touch tilt continues at "
        "15-60s; when the two oppose (product negative), the velocity is "
        "decelerating and a turning point is forming. Economically distinct "
        "from oir_zvel_extreme_15s (velocity x |level|) and "
        "oir_zaccel_extreme_15s (acceleration x |level|): those weight ONE "
        "derivative by the LEVEL extremity; here BOTH derivatives are "
        "regime-normalized against their own histories and crossed, so the "
        "object measures derivative-ON-derivative co-movement, not "
        "derivative-versus-level. A steady high velocity with zero "
        "acceleration scores ~0 here (d2z~0) but maximally under the level-"
        "weighted velocity product."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6-B family brief: velocity-vs-acceleration disagreement "
        "and co-movement on the proven book bases; the agreement product is "
        "the continuous counterpart to the gated turning-point detector."
    ),
    compute=compute,
)
