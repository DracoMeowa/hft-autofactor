"""Explore-lane prototype spec (iter-003 R4, family R4-C).

fullbook_imb_z_cross_vel_15s: z-level vs instantaneous-velocity divergence
on the full-book imbalance -- broad-book regime-SWITCH events: the 300s z
of (total_bid_vol - total_ask_vol)/(sum) crossed zero within the last 15s;
value is the z-velocity, only at crossings, else 0.
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


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the broad-book regime flip.
    """
    z = _z(_fullbook_imb(), W)
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
    name="fullbook_imb_z_cross_vel_15s",
    mechanism=(
        "Broad-book liquidity regime flip events: the trailing-300s z of "
        "the whole-book bid/ask volume ratio crosses zero within 15s. The "
        "full book aggregates patient deeper-queue positioning (ETF "
        "creation/redemption parking, institutional limit interest), which "
        "normally moves slowly; a sign reversal inside 15s means broad "
        "liquidity is actively relocating across the book right now -- an "
        "unusually costly, likely informed repositioning whose new "
        "direction continues at 15-60s. The crossing VELOCITY scores the "
        "decisiveness of the relocation. DEDUP: a different economic "
        "object from library fullbook_imb_z_300s, which is the pure LEVEL "
        "z active every row (regime STATE); here only the transition "
        "EVENTS are scored (event-sparse, 0 elsewhere) -- regime change "
        "vs regime state. Also distinct from library fullbook_imb_mom_60s "
        "(raw 60s delta of the unnormalized ratio)."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R4-C family brief: generalize the admitted "
        "ofi_z_cross_vel_15s crossing template to the full-book "
        "imbalance; round-2 finding that full-book aggregates carry "
        "patient-positioning information motivates the base choice."
    ),
    compute=compute,
)
