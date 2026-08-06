"""Explore-lane prototype spec (iter-003 R4, family R4-C).

ti60_z_cross_vel_15s: z-level vs instantaneous-velocity divergence on
trade_imbalance_60s -- aggressive-flow regime-SWITCH events: the 300s z of
trade_imbalance_60s crossed zero within the last 15s; value is the
z-velocity, only at crossings, else 0.
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
    the signed velocity of the aggressive-flow regime flip.
    """
    z = _z(pl.col("trade_imbalance_60s"), W)
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
    name="ti60_z_cross_vel_15s",
    mechanism=(
        "Aggressive-flow regime flip events: the trailing-300s z of "
        "trade_imbalance_60s (signed marketable volume balance) crosses "
        "zero within 15s. The trade channel is where urgency is revealed: "
        "a minutes-old aggression regime switching sign inside 15s means "
        "the side hitting the book has changed -- informed re-direction "
        "of executable flow rather than queue churn. The crossing "
        "VELOCITY scores how decisive the hand-off is; decisive flips "
        "continue in the new direction at 15-60s (impact of freshly "
        "committed marketable flow is front-loaded). Event-sparse (0 off "
        "crossings), so it differs from the library trade-channel "
        "accumulations (ti_accum_300s, ti_ewm_state_300s: slow state "
        "built every row) and from the dead bare ti z-levels: only the "
        "regime-reversal EVENT is scored, mirroring the admitted "
        "ofi_z_cross_vel_15s on the book channel."
    ),
    info_set="trade_imbalance_60s",
    inspiration=(
        "iter-003 R4-C family brief: generalize the admitted "
        "ofi_z_cross_vel_15s crossing template to the trade-channel "
        "imbalance state column (the cross-channel twin of the OFI "
        "original)."
    ),
    compute=compute,
)
