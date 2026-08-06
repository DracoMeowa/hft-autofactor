"""Explore-lane prototype spec (iter-003 R6, family R6B).

wdi_zjerk_extreme_15s: JERK-extremity product on the 5-level depth
imbalance. The 3rd difference of z (15s jerk = diff of acceleration),
weighted by level extremity |z|. One derivative above the admitted
wdi_zaccel_extreme_15s: abrupt changes in the acceleration of the multi-
level queue rebuild.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity, acceleration, jerk


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d3z * |z| where d3z = 15s jerk of z (3rd derivative).

    Warm-up rows null.
    """
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    d3z = d2z - d2z.shift(LAG)
    return part.select((d3z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zjerk_extreme_15s",
    mechanism=(
        "Jerk-weighted depth-regime stretch: the 15s jerk of z_300(wdi) -- "
        "the 3rd difference, the change rate of the acceleration -- "
        "weighted by how extreme the regime being jerked is (|z|). The "
        "5-level depth imbalance rebuild is costlier and more committed "
        "than touch reposting; its acceleration measures whether the queue "
        "rebuild is speeding up, and its JERK measures whether that "
        "acceleration just abruptly changed. A sudden curvature break in a "
        "multi-level queue regime -- from intensifying to snapping back, or "
        "vice versa -- is the fingerprint of informed repositioning "
        "reversing course, and when it lands on an already-stretched depth "
        "tilt (high |z|) the break is decisive for 15-60s returns. Distinct "
        "from the admitted wdi_zaccel_extreme_15s (d2z * |z|): acceleration "
        "measures intensification; jerk measures abrupt CHANGE in "
        "intensification. A steadily accelerating extreme regime (constant "
        "d2z) reads ~0 here but maximally under zaccel; only regimes whose "
        "curvature is breaking fire."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6-B family brief: JERK (3rd derivative) extremeness on "
        "wdi; the multi-level queue's committed nature makes an abrupt "
        "curvature break a higher-conviction signal than at the touch."
    ),
    compute=compute,
)
