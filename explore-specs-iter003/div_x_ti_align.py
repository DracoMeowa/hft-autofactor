"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

div_x_ti_align: top-5 vs full-book divergence z x trade imbalance -- does
the structural touch-vs-queue mismatch AGREE with the aggressive flow
direction (hidden-absorption / confirmation hypothesis).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
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
    """z(wdi - full-book imbalance, 300s) x trade_imbalance_60s."""
    div_z = _z(pl.col("wdi") - _fullbook_imb(), W)
    ti = pl.col("trade_imbalance_60s")
    return part.select((div_z * ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="div_x_ti_align",
    mechanism=(
        "Structural-flow confirmation: the touch-vs-full-book divergence "
        "measures whether the executable tip is more one-sided than the deep "
        "queue; trade imbalance measures who is actually aggressing. When "
        "the two AGREE in sign (positive product), aggressive flow is "
        "consuming the side of the book that the deep queue also favors -- a "
        "hidden-absorption regime where resting depth validates the flow and "
        "the move tends to persist at 15-60s. When they DISAGREE, a touch "
        "bias is being attacked from the opposite direction without deep "
        "queue support -- fragile, more likely to revert. Multiplying the "
        "divergence z by trade_imbalance_60s turns a static mismatch regime "
        "into a flow-conditioned directional signal; the product is "
        "sign-symmetric and near-orthogonal to either parent alone."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol, trade_imbalance_60s (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 4 (divergence x trade-imbalance "
        "direction, hidden absorption hypothesis); state-conditioned "
        "interactions passed 15s in round 1; top5_book_div_z_300s was "
        "round-2's strongest IC."
    ),
    compute=compute,
)
