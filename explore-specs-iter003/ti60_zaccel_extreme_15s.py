"""Explore-lane prototype spec (iter-003 R6A, family R6A).

ti60_zaccel_extreme_15s: z-ACCELERATION-extremeness product on the 60s
trade imbalance. The 15s acceleration (2nd difference) of the 300s
z-regime of trade_imbalance_60s, weighted by the regime's level extremity
|z|. Mirrors the round-5 winning oir_zaccel template on the aggressive-
flow base.
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
    """d2z * |z| where d2z = 15s acceleration of the trade-imbalance z.

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(pl.col("trade_imbalance_60s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti60_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted aggressive-flow regime stretch: the 15s "
        "acceleration (2nd difference) of z_300(trade_imbalance_60s), "
        "weighted by how stretched that regime is (|z|). "
        "trade_imbalance_60s is the signed balance of aggressive volume "
        "over the trailing minute -- the taker footprint. Its z-accel "
        "isolates INTENSIFYING one-sided aggression from steady-state "
        "imbalance: when the taker-flow regime is already stretched "
        "(high |z|: persistent one-sided aggression beyond the 300s norm) "
        "and its curvature is accelerating further, the aggressive "
        "program is ramping up -- fresh informed flow whose impact is "
        "front-loaded and continues at 15-60s before the book absorbs. "
        "Economically distinct from the round-4 ti60_z_cross_vel_15s "
        "(z sign-flip EVENT scored by velocity): that fires on direction "
        "reversals of the velocity; this scores the acceleration MAGNITUDE "
        "scaled by the level stretch -- a continuous curvature signal. A "
        "steady high-velocity taker regime scores ~0 here (constant dz -> "
        "d2z~0); only regimes whose rate of aggression is itself changing "
        "fire."
    ),
    info_set="trade_imbalance_60s",
    inspiration=(
        "iter-003 R6A family brief: z-acceleration-extremeness template "
        "applied to the trade-imbalance base NOT yet covered in "
        "acceleration form."
    ),
    compute=compute,
)
