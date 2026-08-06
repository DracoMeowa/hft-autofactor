"""Explore-lane prototype spec (iter-003 R5, family R5-B).

ti60_vel15_x_vel60: NEW construction -- cross-timescale velocity mismatch
on trade_imbalance_60s. 15s z-velocity signed by 60s z-velocity direction.
Tests whether aggressive-flow velocity agreement across 15s and 60s
horizons predicts continuation of the taker imbalance.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 300s trailing z window
LAG15 = 5   # 15s fast velocity
LAG60 = 20  # 60s slow velocity


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz15 * sign(dz60) on the trade-imbalance z-regime.

    Warm-up rows null (z warm-up propagates through the 60s shift).
    """
    z = _z(pl.col("trade_imbalance_60s"), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    return part.select((dz15 * dz60.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti60_vel15_x_vel60",
    mechanism=(
        "Cross-timescale velocity confirmation on taker flow: the 15s "
        "z-velocity of z_300(trade_imbalance_60s) multiplied by sign of "
        "the 60s z-velocity. trade_imbalance_60s measures signed "
        "aggressive volume balance at the touch; when both its 15s and "
        "60s z-velocities point the same way, aggressive flow is "
        "sustained across horizons -- takers are hitting one side with "
        "both immediate (15s) and persistent (60s) urgency, the signature "
        "of informed directional trading whose pressure propagates through "
        "price at 15-60s. Disagreement (fast taker flow fighting the slow "
        "taker trend) flips sign, encoding that the slower taker direction "
        "dominates. Distinct from library ti_accel_15_60 (15s-minus-60s "
        "raw acceleration, unnormalized) and ti60_z_cross_vel_15s (round-4 "
        "sign-flip event): this uses cross-horizon velocity AGREEMENT as "
        "the economic input, a multi-timescale confirmation object."
    ),
    info_set="trade_imbalance_60s",
    inspiration=(
        "iter-003 R5-B family brief: cross-timescale velocity mismatch "
        "on the taker-imbalance base; taker flow agreement across 15s/"
        "60s isolates sustained informed aggression."
    ),
    compute=compute,
)
