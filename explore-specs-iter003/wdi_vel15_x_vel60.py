"""Explore-lane prototype spec (iter-003 R5, family R5-B).

wdi_vel15_x_vel60: NEW construction variant of the z-vs-velocity template
on wdi -- CROSS-TIMESCALE velocity mismatch. The 15s velocity of the
z-regime signed by the 60s velocity direction. Tests whether fast and
slow quoting-flow velocity AGREE in direction (multi-timescale confirmation
predicts continuation) or DISAGREE (conflict predicts the slow direction).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing z window
LAG15 = 5   # 5 x 3s = 15s fast velocity
LAG60 = 20  # 20 x 3s = 60s slow velocity


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz15 * sign(dz60) where dz15 = 15s z-velocity, dz60 = 60s z-velocity.

    Warm-up rows null (z warm-up propagates through the 60s shift).
    """
    z = _z(pl.col("wdi"), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    return part.select((dz15 * dz60.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_vel15_x_vel60",
    mechanism=(
        "Cross-timescale velocity confirmation: the 15s z-velocity of "
        "z_300(wdi) multiplied by sign of the 60s z-velocity. When both "
        "timescales of depth-imbalance motion agree (build-up or decay "
        "visible at both 15s and 60s horizons), multi-timescale quoting "
        "flow is aligned -- institutional repositioning spanning minutes, "
        "not just a fleeting quote flicker -- and the fast velocity "
        "magnitude carries continuation signal at 15-60s. When they "
        "disagree (fast pushing one way while slow pushes the other), the "
        "sign flip encodes fast-vs-slow conflict: the slow regime's "
        "direction typically wins as the fast push is absorbed. Distinct "
        "from the round-4 wdi_z_cross_vel_15s (single-timescale sign-flip "
        "event): this never gates to zero, and the economic input is the "
        "AGREEMENT between two lookback horizons of the same derivative."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R5-B family brief: cross-timescale velocity mismatch "
        "construction; 15s/60s agreement isolates multi-horizon flow "
        "alignment that single-window velocity cannot see."
    ),
    compute=compute,
)
