"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

ofi_concord_flip_gate: cross-window OFI conviction x SIGN of the spread-
state z -- regime KIND flips the signal, stress magnitude does not scale it.
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
    """ofi_concord base x sign(z(spread,300s)); warm-up null."""
    base = _ofi_concord()
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((base * sp_z.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_concord_flip_gate",
    mechanism=(
        "Regime-KIND gating of flow conviction, as opposed to regime-"
        "INTENSITY gating: the conviction base keeps its full magnitude and "
        "is flipped purely by the SIGN of the spread-state z. Hypothesis: "
        "what separates informed queue investment from routine churn is the "
        "TYPE of quoting environment, not how deep the stress is -- any "
        "wide-stress episode (z>0) makes persistent same-sign flow an "
        "informed-positioning read (continuation), any tight-comfort "
        "episode (z<0) makes it cheap churn whose direction fades; rows "
        "with exactly neutral quoting (z=0, e.g. long constant-spread "
        "stretches) get zero weight. The product form by contrast nulls "
        "near-neutral rows and overweights extreme stress bursts; this "
        "spec tests whether the information lives in the discrete regime "
        "classification instead. Because |value| == |base|, the magnitude "
        "distribution is identical to the base's -- the economic input "
        "that changes is the regime-conditional SIGN, which is why this "
        "is a distinct question, not a rescaling."
    ),
    info_set="ofi_15s, ofi_60s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: sign-asymmetric-gate variant; "
        "motivated by the round-3 finding that spread-z is the only live "
        "interaction dimension -- here we ask whether the gating acts "
        "through regime kind (sign) rather than stress intensity "
        "(magnitude), which the div_z_x_spread_z-style product assumes."
    ),
    compute=compute,
)
