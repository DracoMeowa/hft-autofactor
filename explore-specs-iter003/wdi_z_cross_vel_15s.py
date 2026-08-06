"""Explore-lane prototype spec (iter-003 R4, family R4-C).

wdi_z_cross_vel_15s: z-level vs instantaneous-velocity divergence on wdi --
fresh depth-imbalance regime-SWITCH events: the 300s z of wdi crossed zero
within the last 15s; value is the z-velocity, only at crossings, else 0.
Direct generalization of the admitted ofi_z_cross_vel_15s template.
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

    Warm-up rows null (z warm-up propagates through the shift and the flip
    indicator); non-crossing rows are exactly 0, crossing rows carry the
    signed velocity of the regime change.
    """
    z = _z(pl.col("wdi"), W)
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
    name="wdi_z_cross_vel_15s",
    mechanism=(
        "Depth-weighted book regime flips: when the trailing-300s z of wdi "
        "(exp-decay weighted 5-level depth imbalance) crosses zero within "
        "the last 15s, the whole visible bid/ask stack has just rebuilt "
        "against an imbalance tilt that had persisted for minutes. The "
        "VELOCITY of the crossing (z now minus z 15s ago, signed by the "
        "new direction) measures the decisiveness of the rebuild: pulling "
        "and refilling quotes across several price levels is costly, so a "
        "fast decisive flip is typically informed repositioning whose new "
        "direction continues at 15-60s, while a slow drift across zero "
        "scores weak. Event-sparse (exactly 0 except at crossings), so it "
        "is a different object from the library raw-delta momenta "
        "(wdi_mom_30s/90s/180s, wdi_accel_90s are unnormalized deltas "
        "active every row) and from the dead bare wdi level-z of round 1: "
        "only the flip event itself is scored here."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R4-C family brief: generalize the admitted "
        "ofi_z_cross_vel_15s template (z-level vs own instantaneous "
        "velocity divergence, passes 15/30/300/900s) to the depth-"
        "weighted book-imbalance state column."
    ),
    compute=compute,
)
