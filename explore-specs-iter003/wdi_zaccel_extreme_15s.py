"""Explore-lane prototype spec (iter-003 R5, family R5-B).

wdi_zaccel_extreme_15s: NEW construction variant of the z-vs-velocity
template on the winning wdi base -- ACCELERATION-extremity product.
The 2nd difference of z (15s acceleration of the z-regime) weighted by
the level extremity |z|. Tests whether acceleration of an already-stretched
depth-imbalance regime carries short-horizon continuation signal --
distinct from the round-4 velocity-extremity product (dz * |z|), which
measures first-derivative motion, not second-derivative intensification.
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
    """d2z * |z| where d2z = 15s acceleration of z (diff of diff of z).

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)       # 15s velocity of z
    d2z = dz - dz.shift(LAG)    # 15s acceleration of z
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted regime stretch: the 15s acceleration of "
        "z_300(wdi) -- the 2nd difference, measuring whether the depth-"
        "imbalance regime's rate of change is ITSELF increasing -- "
        "weighted by how stretched the regime being accelerated is (|z|). "
        "A crowded multi-level queue state (high |z|) whose rebuild speed "
        "is accelerating (d2z pointing further in the regime direction) "
        "marks intensifying informed repositioning: posting and pulling "
        "depth across several levels at INCREASING speed is costlier than "
        "steady rebuilding and more likely committed. The direction of "
        "acceleration continues at 15-60s. Economically distinct from the "
        "round-4 admitted wdi_zvel_extreme_15s (dz * |z|): that measures "
        "velocity -- steady fast motion of an extreme regime; this measures "
        "ACCELERATION -- the curvature, whether the motion is intensifying "
        "or decelerating. An extreme regime can have high velocity but zero "
        "acceleration (steady drift); this scores only the curvature."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R5-B family brief: NEW z-vs-acceleration construction on "
        "the winning wdi base; the round-4 z-vs-velocity products were the "
        "strongest signals ever measured here, and the 2nd derivative is a "
        "genuinely different economic question (intensification vs steady "
        "motion)."
    ),
    compute=compute,
)
