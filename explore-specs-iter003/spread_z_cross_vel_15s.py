"""Explore-lane prototype spec (iter-003 R4, family R4-C).

spread_z_cross_vel_15s: z-level vs instantaneous-velocity divergence on
quoted_spread_ticks -- quoting-cost regime-SWITCH events: the 300s z of the
spread crossed zero within the last 15s; value is the z-velocity, only at
crossings, else 0.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the quoting-cost regime flip.
    """
    z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((flip * (z - z_lag)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="spread_z_cross_vel_15s",
    mechanism=(
        "Quoting-cost regime flip events: the trailing-300s z of the "
        "quoted spread crosses zero within 15s. The spread regime is the "
        "market-makers' risk gauge, and its fast sign flips are "
        "asymmetric in price impact: a decisive widening flip (positive "
        "crossing velocity) marks makers pulling quotes ahead of adverse "
        "information -- fear arriving in seconds -- which tends to be "
        "followed by further downside pressure at 15-60s, while a "
        "decisive tightening flip marks liquidity restoration and "
        "stabilization. Scoring only the crossing VELOCITY makes the "
        "factor event-sparse (0 off flips) and different from the "
        "built-in spread_z_300s level (panel column, IS-dead bare in "
        "round 1) and from the dead spread_z_60s/120s window swaps: the "
        "economic question is the transition event, not the state level. "
        "Round-1/3 lesson: spread-z is the live interaction dimension, "
        "but only in conditioned/derivative forms."
    ),
    info_set="quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-C family brief: generalize the admitted "
        "ofi_z_cross_vel_15s crossing template to the quoting-cost state "
        "column; the round-3 finding that spread-z conditioning is the "
        "only live interaction dimension motivates spread as a base."
    ),
    compute=compute,
)
