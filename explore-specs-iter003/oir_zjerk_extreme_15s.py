"""Explore-lane prototype spec (iter-003 R6, family R6B).

oir_zjerk_extreme_15s: JERK-extremity product on the top-of-book imbalance.
The 3rd difference of z (15s jerk of the z-regime = diff of acceleration),
weighted by level extremity |z|. One derivative above the admitted
oir_zaccel_extreme_15s: captures abrupt changes in the acceleration of
queue repositioning at the touch.
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
    """d3z * |z| where d3z = 15s jerk of z (diff of the acceleration).

    Warm-up rows null (z warm-up propagates through three shifts).
    """
    z = _z(pl.col("oir"), W)
    dz = z - z.shift(LAG)        # 15s velocity of z
    d2z = dz - dz.shift(LAG)     # 15s acceleration of z
    d3z = d2z - d2z.shift(LAG)   # 15s jerk of z (3rd derivative)
    return part.select((d3z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zjerk_extreme_15s",
    mechanism=(
        "Jerk-weighted touch-regime stretch: the 15s jerk of z_300(oir) -- "
        "the 3rd difference, the change rate of the acceleration -- weighted "
        "by how extreme the regime being jerked is (|z|). The best-quote "
        "imbalance is the most actively reposted slot; its acceleration "
        "(2nd derivative) measures whether the rebuild is speeding up, and "
        "its JERK measures whether that acceleration is ITSELF abruptly "
        "changing -- a sudden spike in the rate of intensification (or a "
        "sudden snap from intensifying to decelerating) marks a regime "
        "where market makers just changed their quoting behavior at the "
        "touch. When this happens against an already-stretched touch "
        "regime (high |z|), the abrupt curvature change is decisive: the "
        "queue tilt's regime just broke and price follows the new "
        "direction at 15-60s. Distinct from the admitted "
        "oir_zaccel_extreme_15s (d2z * |z|, the 2nd derivative): that "
        "measures whether acceleration is intensifying; jerk measures "
        "whether the acceleration JUST CHANGED -- a steady extreme "
        "acceleration scores ~0 here (constant d2z -> d3z~0) but maximally "
        "under zaccel; only curvature BREAKS fire here."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6-B family brief: JERK (3rd derivative) extremeness on "
        "the proven bases; round-5 found acceleration (2nd derivative) was "
        "the strongest signal, so the next derivative notch tests whether "
        "abrupt curvature-change carries incremental short-horizon signal."
    ),
    compute=compute,
)
