"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

slope_gated_ofi_300s: order flow let through ONLY in the thin-book regime
-- aggressive book-flow deltas matter for price when the depth profile is
unusually flat, and are absorbed when it is steep.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 300s) when z(book_slope, 300s) < 0, else 0; warm-up null."""
    slope_z = _z(pl.col("book_slope"), W)
    ofi_z = _z(pl.col("ofi_60s"), W)
    gated = (
        pl.when(slope_z.is_null() | ofi_z.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(slope_z < 0.0)
        .then(ofi_z)
        .otherwise(0.0)
    )
    return part.select(gated.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="slope_gated_ofi_300s",
    mechanism=(
        "State-gated flow: the price impact of order-book delta flow is "
        "not constant -- it depends on the depth profile that absorbs it. "
        "book_slope measures how fast depth accumulates away from the "
        "touch; when the shape is unusually FLAT vs its own trailing-300s "
        "regime (slope z < 0), the walls of inventory behind the touch are "
        "thin and a given book-flow delta has OUTSIZED price impact "
        "(continuation in the flow direction). In the unusually steep/"
        "thick regime the same flow is absorbed by stacked depth and "
        "carries little drift. The factor therefore passes the ofi z "
        "through only in the thin-shape regime and is silent elsewhere -- "
        "an impact-state gate, structurally different from a symmetric z x "
        "z product (slope_x_ti_absorb_300s probes the absorption leg with "
        "executed aggression; this one isolates the amplification leg "
        "with book-delta flow). Conditioned flow beats bare flow: raw ofi "
        "variants keep dying on the panel wall while conditioned forms "
        "(ofi_z_x_spread_z) live."
    ),
    info_set="book_slope, ofi_60s",
    inspiration=(
        "iter-003 R4-D brief direction (b) slope x flow interactions; "
        "round-1/round-3 lesson that conditions > levels and spread-z "
        "gating is productive -- here the gate is the SHAPE regime instead "
        "of spread, on a quote-shape column owned by R4-D."
    ),
    compute=compute,
)
