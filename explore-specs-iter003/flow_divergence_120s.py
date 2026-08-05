"""Explore-lane prototype spec (iter-003, flow-interaction lens).

flow_divergence_120s: medium-window (120s z) variant of the champion
flow_divergence_300s -- halfway between the fresh 60s and the slow 300s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 120s) - z(trade_imbalance_60s, 120s)."""
    ofi = pl.col("ofi_60s")
    ti = pl.col("trade_imbalance_60s")
    return part.select((_z(ofi, W) - _z(ti, W)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="flow_divergence_120s",
    mechanism=(
        "Medium-horizon absorption divergence: between the fresh 60s "
        "variant and the champion's 300s window lies the 120s scale, "
        "where a stealth episode that started one or two minutes ago is "
        "still active but no longer noise-dominated. z(ofi) minus z(trade "
        "imbalance) over 120s measures whether book building over the "
        "last two minutes has systematically outrun (positive: passive "
        "bid accumulation beyond what trades justify -> up-continuation) "
        "or lagged (negative: hidden distribution) executed aggression. "
        "The three windowed variants (60/120/300s) triangulate which "
        "absorption bandwidth each prediction horizon rewards."
    ),
    info_set="ofi_60s, trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 6: horizon-matched variants of the "
        "flow_divergence_300s champion (passed 15/30/60s OOS on the "
        "2026-08-05 re-screen); CKS (2014); stealth-limit-flow literature."
    ),
    compute=compute,
)
