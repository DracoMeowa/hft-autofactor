"""Explore-lane prototype spec (iter-003 R4, family R4-C).

oir_z_cross_vel_15s: z-level vs instantaneous-velocity divergence on oir --
fresh touch-queue regime-SWITCH events: the 300s z of oir crossed zero
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

    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the touch-regime flip.
    """
    z = _z(pl.col("oir"), W)
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
    name="oir_z_cross_vel_15s",
    mechanism=(
        "Touch-queue regime flip events: when the trailing-300s z of oir "
        "(best-bid vs best-ask queue ratio) crosses zero within 15s, "
        "control of the touch queue has just changed hands. OIR is the "
        "fastest-moving book state, so a minutes-old touch regime dying "
        "inside 15s marks an abrupt queue pull-and-refill at the best "
        "quotes -- the cheapest place to signal urgency, and the one "
        "informed traders touch first. The crossing VELOCITY scores the "
        "decisiveness of the hand-off; decisive flips continue in the new "
        "direction at 15-60s. Event-sparse (exactly 0 off crossings), so "
        "the economic input differs from library oir_mom_60s (raw 60s oir "
        "delta, active every row): here only regime sign-reversal events "
        "are scored, not ongoing momentum."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R4-C family brief: generalize the admitted "
        "ofi_z_cross_vel_15s z-vs-velocity crossing template to the "
        "top-of-book imbalance state column."
    ),
    compute=compute,
)
