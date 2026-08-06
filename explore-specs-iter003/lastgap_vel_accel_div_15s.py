"""Explore-lane prototype spec (iter-003 R6, family R6B).

lastgap_vel_accel_div_15s: velocity-over-acceleration DOMINANCE ratio on
the last-trade-to-mid gap (ticks). z(15s z-velocity) / (1 + |z(15s z-accel)|):
fires when aggressor-flow velocity is extreme but its acceleration has faded
-- a one-sided aggressor burst that has lost its intensification.
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
    """z(dz) / (1 + |z(d2z)|): velocity-dominance over faded acceleration.

    Bounded ratio (denominator >= 1.0); warm-up rows null.
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
    return tmp.select(
        (pl.col("_zdzz") / (1.0 + pl.col("_zd2zz").abs())).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="lastgap_vel_accel_div_15s",
    mechanism=(
        "Aggressor-burst overextension vs curvature fuel: the z of the 15s "
        "z-velocity of the last-mid gap (ticks) divided by (1 + |z of its "
        "15s z-acceleration|). The aggressor gap is the fastest directional "
        "signal; a sustained one-sided burst registers as an extreme "
        "velocity, but when the acceleration that was driving it has faded "
        "(denominator ~1), the burst has lost its intensification -- the "
        "aggressor side hit hard and then stopped accelerating, the "
        "signature of an exhausting impact thrust that reverts as the book "
        "reforms. When acceleration remains strong the denominator grows, "
        "demoting the still-intensifying aggressor (which the agreement "
        "product captures as continuation). Sign follows velocity, so the "
        "exhaustion direction is preserved. Distinct from "
        "lastgap_zvel_x_zaccel_15s (multiplies: acceleration amplifies, "
        "capturing continuation) and from lastgap_zvel_extreme_15s (velocity "
        "x level): acceleration in the DENOMINATOR captures the opposite "
        "regime -- aggressor velocity outrunning dead curvature, the "
        "exhaustion/reversion signal."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R6-B family brief: velocity/acceleration ratio on the "
        "aggressor gap; the fastest base should show the cleanest "
        "acceleration-fade exhaustion signal."
    ),
    compute=compute,
)
