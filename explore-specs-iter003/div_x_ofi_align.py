"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

div_x_ofi_align: top-5 vs full-book divergence z x OFI z -- does the
structural touch-vs-queue mismatch AGREE with the order-book-delta flow
(book-side confirmation of the mismatch).
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
    """z(wdi - full-book imbalance, 300s) x z(ofi_60s, 300s)."""
    div_z = _z(pl.col("wdi") - _fullbook_imb(), W)
    ofi_z = _z(pl.col("ofi_60s"), W)
    return part.select((div_z * ofi_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="div_x_ofi_align",
    mechanism=(
        "Book-flow confirmation of the structural mismatch: OFI is the "
        "order-book-delta flow -- where resting orders are being added or "
        "pulled right now. When the touch-vs-full-book divergence and OFI "
        "point the SAME way (positive product), the queue is actively "
        "REBUILDING on the side the deep book already favors: fresh resting "
        "interest arriving to back the mismatch, a self-reinforcing regime "
        "that continues at 15-60s. When they oppose, the visible mismatch is "
        "being actively withdrawn (orders pulled from the favored side), a "
        "hollowing-out that precedes reversion. OFI is the natural flow "
        "companion for a BOOK-structure mismatch (both are resting-order "
        "phenomena), complementing div_x_ti_align which uses executed "
        "aggression; z-scoring both parents regime-normalizes the product."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol, ofi_60s (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 4 (divergence x ofi_60s direction); "
        "order-book-delta flow (Cont, Kukanov & Stoikov 2014); state-"
        "conditioned interactions passed 15s in round 1."
    ),
    compute=compute,
)
