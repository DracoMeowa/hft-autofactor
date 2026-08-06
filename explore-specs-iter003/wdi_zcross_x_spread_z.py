"""Explore-lane prototype spec (iter-003 R5, family R5-C).

wdi_zcross_x_spread_z: depth-imbalance regime-flip velocity gated by
spread-state stress -- the admitted wdi_z_cross_vel_15s base (z of wdi
crossed zero within 15s; value is the crossing velocity) multiplied by
the 300s spread z. Tests whether whole-stack depth rebuilds predict
longer-horizon returns only when spreads are stressed.
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
    """crossing velocity of wdi * z(spread, 300s); warm-up null."""
    z = _z(pl.col("wdi"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    cross_vel = flip * (z - z_lag)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((cross_vel * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zcross_x_spread_z",
    mechanism=(
        "Spread-stress-gated depth-imbalance regime flips: wdi_z_cross_vel_"
        "15s (the admitted crossing template on the exp-decay weighted "
        "5-level depth imbalance) scores sign-reversal events where the "
        "whole visible bid/ask stack rebuilds against a tilt that persisted "
        "for minutes. Such a rebuild is costly -- pulling and refilling "
        "across five price levels -- and hypothesis: it is only worth that "
        "cost when it is informed. But under TIGHT spreads the rebuild can "
        "be cheap routine repositioning; under WIDE spreads the cost is "
        "real and the flip is credible as informed repositioning. "
        "Multiplying the crossing velocity by z(spread, 300s) re-weights "
        "each stack-rebuild event by the spread-stress regime, isolating "
        "the costly informed flips that continue at 300-900s while "
        "suppressing the cheap routine ones. This is a different economic "
        "question from wdi_zvel_x_spread_z (same family): that gates the "
        "CONTINUOUS extremity-weighted velocity; this gates the EVENT of a "
        "regime sign-reversal -- a stack rebuild, not ongoing momentum."
    ),
    info_set="wdi, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-C family brief: condition the round-4 z-velocity "
        "winner wdi_z_cross_vel_15s on spread-z; the brief asks to spread "
        "specs across both the extreme-velocity and crossing-velocity "
        "forms of the wdi base."
    ),
    compute=compute,
)
