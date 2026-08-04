"""Explore-lane prototype spec (iter-001, flow-queue lens).

flow_divergence_300s: absorption / stealth-limit-flow signal.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z300(x: pl.Expr) -> pl.Expr:
    """Trailing-300s z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=W, min_samples=W)
    s = x.rolling_std(window_size=W, min_samples=W)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    ofi = pl.col("ofi_60s")
    ti = pl.col("trade_imbalance_60s")
    return part.select((_z300(ofi) - _z300(ti)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="flow_divergence_300s",
    mechanism=(
        "Absorption / stealth limit flow: the trailing-300s z-score of OFI "
        "(order-book-delta flow: limit placement/pull at the touch) minus the "
        "z-score of executed trade imbalance. Positive divergence means the "
        "book is being built on the bid side relative to what aggressive "
        "trades justify - passive buyers absorbing sell aggression (iceberg/"
        "stealth accumulation); negative divergence is the mirror. The "
        "limit-flow component dominates for next-move prediction (CKS: OFI "
        "R2 > signed volume), and disagreement between executed and "
        "book-building flow flags informed traders hiding in the book, so the "
        "divergence direction should carry into 15s-300s returns."
    ),
    info_set="ofi_60s, trade_imbalance_60s (library)",
    inspiration=(
        "Digest iter-000: ofi vs trade_imbalance rho=0.49 and 'the "
        "ORDER-BOOK-DELTA component of OFI not in trade imbalance is worth "
        "isolating'; Cont-Kukanov-Stoikov (2014) OFI dominates signed volume; "
        "absorption via iceberg orders (Buti & Rindi 2013, undisplayed "
        "liquidity)."
    ),
    compute=compute,
)
