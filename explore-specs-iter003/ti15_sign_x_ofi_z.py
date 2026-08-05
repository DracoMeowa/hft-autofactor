"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ti15_sign_x_ofi_z: cross-channel AND cross-scale agreement -- fast trade
aggression direction (sign of trade_imbalance_15s) re-signing the slow
book-flow regime surprise (300s z of ofi_60s).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window on ofi_60s


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """sign(trade_imbalance_15s) x z(ofi_60s, 300s); warm-up rows null."""
    z_ofi = _z(pl.col("ofi_60s"), W)
    st = pl.col("trade_imbalance_15s").sign()
    return part.select((st * z_ofi).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti15_sign_x_ofi_z",
    mechanism=(
        "Aggressors hitting a book that confirms: the sign of the fastest "
        "trade-channel imbalance (trade_imbalance_15s) re-signs the "
        "trailing-300s z-surprise of book-channel flow (ofi_60s). Large "
        "positive values mean BOTH channels push the same way at their "
        "own scales -- marketable orders arriving while the book regime "
        "is unusually one-sided in the same direction -- the "
        "highest-conviction informed state, continuing at 15-60s. Large "
        "negative values mean aggressors are hitting a book whose flow "
        "regime points the OTHER way: passive queue investment absorbing "
        "aggression (iceberg/stealth behavior), a stall-or-fade warning "
        "scored against the aggressors. This deliberately crosses BOTH "
        "the channel boundary (trade vs book -- partial information "
        "overlap, rho ~ 0.5) AND the scale boundary (15s impulse vs "
        "300s regime), so it is not the dead ofi_ti_agree_60s "
        "(same-window sign share of the two 60s columns), not the "
        "saturated flow_divergence family (z-magnitude GAP of the two "
        "60s columns), and not ofi_concord_15_60 (same-channel "
        "cross-window agreement): agreement here is measured between a "
        "fast trade-side direction and a slow book-side regime."
    ),
    info_set="trade_imbalance_15s, ofi_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R3-C brief direction 2 (cross-scale alignment beyond "
        "existing factors: TRADE-side fast vs BOOK-side flow regime); "
        "ofi_ti_agree_60s rejection showed same-window agreement is "
        "weak; the cross-scale/cross-channel cell is unoccupied."
    ),
    compute=compute,
)
