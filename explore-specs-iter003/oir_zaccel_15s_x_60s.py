"""Explore-lane prototype spec (iter-003 R6D, family R6D).

oir_zaccel_15s_x_60s: cross-timescale ACCELERATION agreement on the
top-of-book imbalance. The 15s z-acceleration of z_300(oir) signed by
the 60s z-acceleration direction, weighted by regime extremity |z|.
Extremeness-weighted per the round-5 mandatory rule.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100    # 100 x 3s = 300s trailing z window
LAG15 = 5   # 5 x 3s = 15s fast lookback
LAG60 = 20  # 20 x 3s = 60s slow lookback


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
    z = _z(pl.col("oir"), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    d2z15 = dz15 - dz15.shift(LAG15)
    d2z60 = dz60 - dz60.shift(LAG60)
    return part.select((d2z15 * d2z60.sign() * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zaccel_15s_x_60s",
    mechanism=(
        "Cross-timescale acceleration confirmation: the 15s z-acceleration "
        "(2nd difference) of z_300(oir) multiplied by the sign of the 60s "
        "z-acceleration, all weighted by regime extremity |z|. oir is the "
        "top-of-book imbalance ratio -- the most snapshot-reactive book "
        "state. When its fast (15s) AND slow (60s) curvature point the same "
        "way, the intensification of one-sided queue rebuilding is confirmed "
        "at both horizons: a 15s acceleration push that agrees with a 60s "
        "acceleration push marks a regime where informed quoting is "
        "ACCELERATING its commitment across multiple timescales, not a "
        "flicker. The |z| weight ensures only already-stretched regimes "
        "contribute -- routine neutral-regime acceleration is suppressed. "
        "Economically distinct from the round-5 wdi_vel15_x_vel60 "
        "(first-derivative cross-timescale agreement): that asks whether "
        "steady velocity is aligned; this asks whether the CURVATURE is "
        "aligned, isolating intensification episodes where the rate of "
        "change is itself increasing on both scales. A regime with aligned "
        "velocity but zero acceleration (steady aligned drift) scores ~0 "
        "here; only regimes whose rebuild speed is accelerating at BOTH "
        "15s and 60s fire."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6D family brief direction 1: cross-timescale "
        "acceleration agreement on book-imbalance bases, extremeness-"
        "weighted per round-5 mandatory rule. Extends the admitted "
        "wdi_vel15_x_vel60 (velocity agreement) to the 2nd derivative."
    ),
    compute=compute,
)
