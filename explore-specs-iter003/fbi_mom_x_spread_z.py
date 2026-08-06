"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

fbi_mom_x_spread_z: 60s momentum of full-book bid/ask volume imbalance
(recomputed fullbook_imb_mom_60s) x spread-state z -- stress-gated
whole-book pressure momentum.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing spread-state window
D = 20    # 20 x 3s rows = 60s momentum window (as in the base)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """fullbook_imb_mom_60s base x z(quoted_spread_ticks, 300s)."""
    base = _fullbook_imb().diff(D)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((base * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fbi_mom_x_spread_z",
    mechanism=(
        "Stress-gated whole-book pressure momentum: fullbook_imb_mom_60s "
        "measures fresh one-sided queue build-up or withdrawal across the "
        "ENTIRE visible book (batch-2 total_*_vol, wider than depth_*5). "
        "When that 60s pressure shift happens under WIDE stressed spreads, "
        "makers are fearful and posting depth is an adverse-selection-"
        "bearing act, so a deliberate whole-book lean toward one side is "
        "informed pre-positioning ahead of impact -> continuation of the "
        "pressure direction at 15-60s. The identical imbalance momentum "
        "under comfortable tight quoting is routine quote maintenance with "
        "little directional content (product flips it). This is the "
        "book-MOMENTUM channel under stress gating, economically distinct "
        "from div_z_x_spread_z (which gates the STRUCTURAL touch-vs-deep "
        "mismatch, a level divergence z, not a 60s pressure change) and "
        "from oir/wdi/depth5 momenta (narrower bases, ungated)."
    ),
    info_set="total_bid_vol, total_ask_vol, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: fullbook_imb_mom_60s (R2-C admitted) "
        "still lacks a spread-z interaction; round-1 meta-lesson that book-"
        "imbalance momenta are the strongest short-horizon cluster + "
        "round-3 that spread-z is the only live interaction dimension."
    ),
    compute=compute,
)
