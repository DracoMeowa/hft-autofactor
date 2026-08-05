"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

div_z_x_spread_z: top-5 vs full-book divergence z x spread-state z -- the
structural mismatch is most informative when quoting itself is stressed.
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
    """z(wdi - full-book imbalance, 300s) x z(quoted_spread_ticks, 300s)."""
    div_z = _z(pl.col("wdi") - _fullbook_imb(), W)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((div_z * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="div_z_x_spread_z",
    mechanism=(
        "Stress-gated structural mismatch: a touch-vs-full-book divergence "
        "under WIDE, stressed spreads is a different animal than the same "
        "divergence under comfortable quoting. Wide spreads mark fearful "
        "market-making and adverse-selection risk, so a book that still "
        "builds a one-sided touch bias against the deep queue in that state "
        "is likely informed positioning rather than routine queue churn -> "
        "stronger continuation of the divergence's implied direction at "
        "15-60s. The identical structure under unusually tight spreads may "
        "be benign noise. Multiplying the divergence z by the spread-state z "
        "gates the mismatch by the quoting regime; the product is sign-"
        "symmetric and near-orthogonal to either parent, and is distinct from "
        "flow_divergence_x_spread_z (which conditions the FLOW divergence, "
        "not the book-structure divergence)."
    ),
    info_set=(
        "wdi, total_bid_vol, total_ask_vol, quoted_spread_ticks (batch-2)"
    ),
    inspiration=(
        "iter-003 R3-B brief direction 4 (divergence x spread_z state); "
        "flow_divergence_x_spread_z passed 15s in round 1 with the same "
        "stress-gating logic applied to a different divergence base."
    ),
    compute=compute,
)
