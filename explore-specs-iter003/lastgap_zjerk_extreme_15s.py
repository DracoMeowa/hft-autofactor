"""Explore-lane prototype spec (iter-003 R6, family R6B).

lastgap_zjerk_extreme_15s: JERK-extremity product on the last-trade-to-mid
gap (ticks). The 3rd difference of z (15s jerk = diff of acceleration),
weighted by level extremity |z|. Abrupt curvature changes in the fastest
aggressor-flow regime -- aggressor intensity snapping direction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity, acceleration, jerk
TICK = 0.001  # SSE ETF min increment


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _lastgap() -> pl.Expr:
    """(last_px - mid_px) / TICK in ticks; positive = buyer aggression."""
    return (pl.col("last_px") - pl.col("mid_px")) / TICK


def compute(part: pl.DataFrame) -> pl.Series:
    """d3z * |z| where d3z = 15s jerk of z (3rd derivative).

    Warm-up rows null.
    """
    z = _z(_lastgap(), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    d3z = d2z - d2z.shift(LAG)
    return part.select((d3z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="lastgap_zjerk_extreme_15s",
    mechanism=(
        "Jerk-weighted aggressor-regime stretch: the 15s jerk of z_300(last-"
        "mid gap in ticks) -- the 3rd difference, the change rate of the "
        "acceleration -- weighted by how extreme the aggressor regime being "
        "jerked is (|z|). The gap is the fastest directional signal; its "
        "acceleration measures whether aggressor intensity is ramping, and "
        "its JERK measures whether that ramp just abruptly broke -- a "
        "one-sided aggressor burst that snaps from intensifying to "
        "decelerating (or the reverse) in 15s. When such a curvature break "
        "lands on an already-stretched aggressor regime (sustained extreme "
        "|z|: a persistent program execution), the break is decisive: the "
        "informed flow just changed its intensity regime and 15-60s returns "
        "follow. Distinct from lastgap_zvel_extreme_15s (velocity x |level|) "
        "and the gap_zaccel form (acceleration x |level|): jerk is the 3rd "
        "derivative, isolating curvature BREAKS in aggressor intensity that "
        "the lower derivatives smooth over -- a steadily-ramping aggressor "
        "burst reads ~0 here but maximally under the acceleration form."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R6-B family brief: JERK (3rd derivative) extremeness on "
        "the aggressor gap; the fastest base should carry the sharpest "
        "curvature-break signal."
    ),
    compute=compute,
)
