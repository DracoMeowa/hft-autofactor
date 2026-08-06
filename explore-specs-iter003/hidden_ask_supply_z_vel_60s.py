"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_ask_supply_z_vel_60s: 60s VELOCITY of the admitted hidden-ask-supply
regime z -- hidden ask liquidity ARRIVING vs WITHDRAWING (supply-side twin).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing z window (matches admitted parent)
LAG = 20  # 20 x 3s rows = 60s velocity lag


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _hidden_ask_share() -> pl.Expr:
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    return (
        pl.when(ta > 0.0)
        .then(ha / ta)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """diff_60s of z(ask-side hidden share, 300s); warm-up rows null."""
    z = _z(_hidden_ask_share(), W)
    return part.select(z.diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_ask_supply_z_vel_60s",
    mechanism=(
        "Arriving vs withdrawing hidden ask supply: the 60s change of the "
        "trailing-300s z of the ask-side hidden share (the admitted "
        "hidden_ask_supply_z_300s regime, differentiated). Hidden supply "
        "ARRIVING (rising z) is patient selling interest freshly stacked "
        "above the touch -- distribution intent queuing out of sight, which "
        "keeps refilling the visible offer and caps rallies / precedes "
        "downward drift. Supply WITHDRAWING (falling z) is the latent "
        "overhang thinning: with little hidden depth left above, consumption "
        "of the visible offer meets no reserve and price walks up more "
        "easily -- an early-strength signal invisible to the level, since a "
        "large but draining overhang and a small but building one are "
        "opposite regimes at the same z-extreme. Supply-side twin of the "
        "bid-support velocity; the two need not mirror each other (hidden "
        "liquidity rotates between sides). Derivative of a z-state in the "
        "live ratio/z class, distinct from the dead raw-hidden-qty momentum "
        "class (momentum of absolute hidden quantities)."
    ),
    info_set="total_ask_vol, depth_ask5 (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (a): velocity of the admitted R3-B "
        "hidden-side z's; hidden_ask_supply_z_300s was a round-3 R3-B "
        "admission -- this differentiates it."
    ),
    compute=compute,
)
