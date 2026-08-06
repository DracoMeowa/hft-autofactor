"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

ofi_concord_wide_gate: cross-window OFI conviction active ONLY when the
spread state is unusually wide (one-sided clip gate) -- a pure
stress-unlocked version with no tight-regime sign-flip claim.
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
    """ofi_concord base x clip(z(spread,300s), 0, inf); warm-up null."""
    base = _ofi_concord()
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    gate = sp_z.clip(lower_bound=0.0)
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_concord_wide_gate",
    mechanism=(
        "Stress-UNLOCKED flow conviction, one-sided claim: cross-window "
        "same-sign OFI is scored only when the quoted spread is unusually "
        "WIDE vs its trailing 300s, and set to exactly zero outside that "
        "regime. The hypothesis is narrower than the product form's: "
        "conviction under stressed quoting is informed queue investment "
        "(makers fearful, yet one-sided limit flow persists across both "
        "windows -> deliberate positioning) and continues at 15-60s, while "
        "conviction under normal or tight quotes makes NO directional "
        "claim at all -- it is switched off rather than claimed to revert. "
        "Economically this isolates the rare informed-positioning episodes "
        "from the mass of routine flow; the nonzero support is a strict "
        "subset of the product's, so its cross-sectional structure (mass "
        "of exact zeros plus stressed-only signals) differs from both the "
        "bare base and ofi_concord_x_spread_z. Dedup note: within the "
        "wide regime the value is monotone in the base, so rank "
        "correlation vs the base is diluted only by the zero mass -- "
        "expect moderate, not negligible, sibling corr with the product."
    ),
    info_set="ofi_15s, ofi_60s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: one-sided-gate variant requested "
        "alongside the plain product; spread-z gating is the only live "
        "interaction dimension after round 3 (3/3 products passed); "
        "ofi_concord_15_60 admitted R2-B without any spread interaction."
    ),
    compute=compute,
)
