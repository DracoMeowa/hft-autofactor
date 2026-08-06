"""Explore-lane prototype spec (iter-003 R6, family R6B).

lastgap_zvel_x_zaccel_15s: velocity-vs-acceleration AGREEMENT product on
the last-trade-to-mid gap (ticks). z(15s z-velocity) crossed with z(15s
z-acceleration) of the aggressor-side regime. The fastest directional
microstructure signal: co-intensifying aggressor velocity + acceleration
marks accelerating informed execution.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity and acceleration
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
    """z(dz) * z(d2z) where dz = 15s z-velocity, d2z = 15s z-acceleration
    of the aggressor-gap z-regime.

    Warm-up rows null.
    """
    z_e = _z(_lastgap(), W)
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
    name="lastgap_zvel_x_zaccel_15s",
    mechanism=(
        "Accelerating informed aggression: the product of the trailing-300s "
        "z of the 15s z-velocity of the last-mid gap (ticks) with the z of "
        "its own 15s z-acceleration. The gap is the fastest directional "
        "microstructure signal -- positive = buyer lifted ask, negative = "
        "seller hit bid. When the aggressor-bias z-velocity and its z-"
        "acceleration co-fire the SAME direction (product large positive), "
        "the aggressor side is not just persistent but INTENSIFYING -- the "
        "rate of one-sided marketable flow is itself accelerating, the "
        "hallmark of an urgency/program execution -- and impact continues "
        "in the aggression direction at 15-60s before the book re-"
        "equilibrates; when they oppose (product negative), the aggression "
        "is decelerating into exhaustion/reversion. Distinct from "
        "lastgap_zvel_extreme_15s (velocity x |level|): here both terms are "
        "derivatives of the gap z, each normalized against its own 300s "
        "distribution, so the object is pure aggressor-flow curvature "
        "co-movement -- a steady-fast aggression with zero acceleration "
        "reads near zero here but maximally under the level-weighted form."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R6-B family brief: velocity-acceleration agreement product "
        "on the aggressor-gap base; the gap is the fastest proven base, so "
        "its acceleration co-movement should carry the sharpest short-"
        "horizon continuation signal."
    ),
    compute=compute,
)
