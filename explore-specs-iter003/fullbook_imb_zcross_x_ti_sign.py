"""Explore-lane prototype spec (iter-003 R5, family R5-C).

fullbook_imb_zcross_x_ti_sign: broad-book regime-flip velocity multiplied
by the sign of trade_imbalance_60s. Tests whether concordance between
patient full-book repositioning and aggressive trade direction predicts
longer-horizon returns.
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
    """fullbook imb crossing velocity * sign(ti_60s); warm-up null."""
    z = _z(_fullbook_imb(), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    cross_vel = flip * (z - z_lag)
    ti_sign = pl.col("trade_imbalance_60s").sign()
    return part.select((cross_vel * ti_sign).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zcross_x_ti_sign",
    mechanism=(
        "Trade-direction concordance of broad-book flips: the full-book "
        "imbalance crossing marks the entire patient book relocating -- "
        "institutional limit interest and ETF inventory shifting sides. "
        "Hypothesis: when this patient repositioning AGREES in direction "
        "with aggressive executed trade flow (ti_60s sign), the same "
        "meta-order is visible in both channels: the institution parks "
        "limit inventory on one side AND executes aggressively in the same "
        "direction -- maximum conviction, and the direction continues at "
        "300-900s. When they DIVERGE (patient book tilts bid but trades "
        "are sell-driven, or vice versa), the institution is providing "
        "liquidity against its own directional interest, which is a "
        "reversion signal. The product (crossing velocity * ti sign) "
        "encodes both: positive on concordance, negative on divergence. "
        "This is the PRODUCT form on the BROADEST book base -- distinct "
        "from oir_zcross_x_ti_sign (GATE form on the touch base) and from "
        "wdi_zvel_x_ofi_sign (book-flow channel on depth): here the "
        "question is whether the WIDEST book scale's patient repositioning "
        "is confirmed by aggressive trades."
    ),
    info_set="total_bid_vol, total_ask_vol, trade_imbalance_60s",
    inspiration=(
        "iter-003 R5-C family brief direction 4: condition z-vel winners "
        "on trade pressure sign; product form on the fullbook imbalance "
        "crossing base to test the widest book-scale concordance."
    ),
    compute=compute,
)
