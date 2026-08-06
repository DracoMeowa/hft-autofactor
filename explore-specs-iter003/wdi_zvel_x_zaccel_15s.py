"""Explore-lane prototype spec (iter-003 R6, family R6B).

wdi_zvel_x_zaccel_15s: velocity-vs-acceleration AGREEMENT product on the
5-level depth imbalance. z(15s z-velocity) crossed with z(15s z-acceleration)
of the multi-level queue regime. Measures whether the depth-imbalance
rebuild is co-intensifying (velocity and acceleration same direction at
co-extreme strength) versus decelerating.
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
    """z(dz) * z(d2z) where dz = 15s z-velocity, d2z = 15s z-acceleration.

    Warm-up rows null (z warm-up propagates through two shifts and two z
    windows).
    """
    z_e = _z(pl.col("wdi"), W)
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
    name="wdi_zvel_x_zaccel_15s",
    mechanism=(
        "Multi-level queue rebuild co-intensification: the product of the "
        "trailing-300s z of the 15s z-velocity of wdi with the z of its own "
        "15s z-acceleration. wdi is the 5-level depth imbalance, a broader "
        "queue state than the touch; rebuilding it across several levels is "
        "costlier than touching the best quote and more likely committed. "
        "When the z-velocity of this regime and its z-acceleration both read "
        "co-extreme the SAME way (product large positive), the queue is "
        "being repositioned with accelerating commitment -- the rebuild is "
        "speeding up, not holding steady -- and the depth tilt continues at "
        "15-60s; when they oppose (product negative), the rebuild is "
        "decelerating into a turn. Distinct from the admitted "
        "wdi_zvel_extreme_15s / wdi_zaccel_extreme_15s (single derivative x "
        "|level|): this crosses BOTH derivatives after each is normalized "
        "against its own 300s distribution, so it scores the curvature-of-"
        "the-motion jointly with the motion -- a regime whose velocity is "
        "extreme but acceleration is mid reads near zero here."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6-B family brief: velocity-acceleration agreement product "
        "on the proven wdi base; round-5 found acceleration-extremity was "
        "the strongest signal, so crossing the two derivatives directly "
        "isolates the co-intensification regime."
    ),
    compute=compute,
)
