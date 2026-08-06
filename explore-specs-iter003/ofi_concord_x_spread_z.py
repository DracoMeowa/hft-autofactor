"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

ofi_concord_x_spread_z: cross-window OFI conviction (the recomputed
ofi_concord_15_60 base) x spread-state z -- stress-gated flow conviction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing spread-state window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _ofi_concord() -> pl.Expr:
    """Recomputed ofi_concord_15_60: sign(ofi_15s) x min(|ofi_15s|,|ofi_60s|)."""
    a = pl.col("ofi_15s")
    b = pl.col("ofi_60s")
    mag = (a.abs() + b.abs() - (a - b).abs()) / 2.0
    return a.sign() * mag


def compute(part: pl.DataFrame) -> pl.Series:
    """ofi_concord base x z(quoted_spread_ticks, 300s); warm-up null."""
    base = _ofi_concord()
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((base * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_concord_x_spread_z",
    mechanism=(
        "Stress-gated cross-window flow conviction: ofi_concord scores "
        "same-sign book flow across the 15s and 60s windows (sign x weaker-"
        "leg magnitude), the queue-investment signature. The SAME conviction "
        "means different things in different quoting regimes. Under WIDE, "
        "stressed spreads market makers are fearful and adverse-selection "
        "risk is high, so one-sided queue investment persisted across both "
        "windows despite that cost is likely informed positioning -> "
        "continuation of the conviction direction at 15-60s. Under "
        "unusually TIGHT comfortable spreads the same persistent flow is "
        "cheap routine two-sided churn whose direction should not carry. "
        "Multiplying by the spread-state z gates magnitude AND flips sign "
        "across regimes; the product is near-orthogonal to the bare base "
        "(round-3 products ran 0.22-0.42 panel corr vs their bases). "
        "Distinct from ofi_z_x_spread_z (round 1): that gated single-window "
        "OFI z-SURPRISE; here the gated input is cross-window AGREEMENT "
        "capped at the weaker leg, a persistence object, not a surprise."
    ),
    info_set="ofi_15s, ofi_60s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: ofi_concord_15_60 (R2-B admitted, 15s "
        "t 10.1) still lacks a spread-z interaction, and spread-z is the "
        "only live interaction dimension (div_z_x_spread_z / dev_open_x_"
        "spread_z / conc_imb_x_spread_z all passed); canonical gate form of "
        "div_z_x_spread_z applied to the conviction base."
    ),
    compute=compute,
)
