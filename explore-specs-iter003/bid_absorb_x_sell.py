"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

bid_absorb_x_sell: hidden BID-support z x (negative) trade imbalance --
aggressive SELLING meeting a deep hidden-bid reservoir: absorption that
blunts downside and precedes a floor / rebound.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _hidden_bid_share() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    return (
        pl.when(tb > 0.0)
        .then(hb / tb)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(hidden_bid/total_bid, 300s) x (-trade_imbalance_60s)."""
    hb_z = _z(_hidden_bid_share(), W)
    ti = pl.col("trade_imbalance_60s")
    return part.select((hb_z * (-ti)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="bid_absorb_x_sell",
    mechanism=(
        "One-sided hidden absorption: the interaction isolates the specific "
        "event of aggressive SELLING (negative trade imbalance) running into "
        "an unusually DEEP hidden-bid reservoir (high z of the bid-side "
        "hidden share). Patient bids parked far below the touch are exactly "
        "the depth that absorbs downside aggression without executing -- the "
        "sells are taken up by the hidden layer, price impact is blunted, "
        "and the move exhausts itself, setting up a floor and upward "
        "reversion/continuation at 15-60s. When hidden bid support is low, "
        "the same selling has nothing beneath it and drives price down. This "
        "is a directional, one-sided absorption bet, not the symmetric "
        "alignment of hidden_imb_x_ti nor the level of hidden_bid_support_z: "
        "it fires only when hidden demand and sell aggression co-occur."
    ),
    info_set="depth_bid5, total_bid_vol, trade_imbalance_60s (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 4/5 (hidden-depth divergence x flow, "
        "hidden absorption hypothesis); absorption/iceberg literature (Buti "
        "& Rindi 2013); interactions passed where bare levels died."
    ),
    compute=compute,
)
