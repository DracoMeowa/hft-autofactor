"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_ask_supply_z_vel_15s: FAST 15s velocity of the admitted hidden-ask-
supply regime z -- abrupt stack/pull events of patient ask depth.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window (matches admitted parent)
LAG = 5  # 5 x 3s rows = 15s velocity lag


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
    """diff_15s of z(ask-side hidden share, 300s); warm-up rows null."""
    z = _z(_hidden_ask_share(), W)
    return part.select(z.diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_ask_supply_z_vel_15s",
    mechanism=(
        "Abrupt hidden ask stack/pull events: the 15s change of the "
        "trailing-300s z of the ask-side hidden share. A sudden surge of "
        "hidden ask supply marks decisive distribution intent being parked "
        "above the touch (informed sellers pre-positioning an overhang), "
        "which caps the next ticks; a sudden collapse of the overhang marks "
        "supply conviction evaporating, freeing the offer to be consumed. "
        "The fast window isolates discrete large-order placement/yank events "
        "from the slower drift of positioning, so the signal is freshest and "
        "should decay within 15-30s -- the horizons where fast book state "
        "carried round-1. Supply-side twin of the bid-support fast velocity; "
        "the two sides rotate independently, so this is not the sign-flip "
        "of its bid twin. Derivative of a z-state (live class), not raw "
        "hidden-qty momentum (dead class)."
    ),
    info_set="total_ask_vol, depth_ask5 (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (a): 15s velocity of the admitted "
        "hidden-side z's; round-1 lesson that 15-30s horizons reward fast "
        "book state; differentiates the admitted R3-B level regime."
    ),
    compute=compute,
)
