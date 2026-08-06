"""Explore-lane prototype spec (iter-003 R6D, family R6D).

wdi_zaccel_15s_x_60s: cross-timescale ACCELERATION agreement on the
5-level depth imbalance. The 15s z-acceleration of z_300(wdi) signed by
the 60s z-acceleration direction, weighted by regime extremity |z|.
Extremeness-weighted per the round-5 mandatory rule.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG15 = 5
LAG60 = 20


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z15 * sign(d2z60) * |z| -- cross-timescale z-accel agreement.

    Warm-up rows null (z warm-up propagates through nested shifts).
    """
    z = _z(pl.col("wdi"), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    d2z15 = dz15 - dz15.shift(LAG15)
    d2z60 = dz60 - dz60.shift(LAG60)
    return part.select((d2z15 * d2z60.sign() * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zaccel_15s_x_60s",
    mechanism=(
        "Cross-timescale acceleration confirmation: the 15s z-acceleration "
        "of z_300(wdi) multiplied by the sign of the 60s z-acceleration, "
        "weighted by regime extremity |z|. wdi is the 5-level depth "
        "imbalance -- passive institutional queue placement across the "
        "visible book. When its fast (15s) AND slow (60s) acceleration "
        "agree in direction, the depth-rebuilding curvature is confirmed "
        "across horizons: limit-order programs are intensifying their "
        "one-sided commitment at increasing speed on both the 15s and "
        "60s scale, not a fleeting flicker. The |z| weight ensures only "
        "already-stretched depth regimes contribute. Economically distinct "
        "from the admitted wdi_vel15_x_vel60 (1st-derivative cross-"
        "timescale agreement): that asks whether steady velocity is "
        "aligned across scales; this asks whether the CURVATURE (rate of "
        "velocity change) is aligned. A regime with aligned velocity but "
        "zero acceleration (steady aligned rebuild) scores ~0 here; only "
        "regimes whose rebuild speed is itself accelerating on both scales "
        "fire. Also distinct from the admitted wdi_zaccel_extreme_15s "
        "(single-timescale 15s acceleration x |z|): that cannot distinguish "
        "a 15s flicker from a multi-minute intensification; this requires "
        "60s corroboration."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6D family brief direction 1: cross-timescale "
        "acceleration agreement. Extends wdi_vel15_x_vel60 (velocity "
        "agreement) and wdi_zaccel_extreme_15s (single-scale accel) to "
        "the multi-scale acceleration product."
    ),
    compute=compute,
)
