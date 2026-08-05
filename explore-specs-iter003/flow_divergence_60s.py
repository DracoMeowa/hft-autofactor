"""Explore-lane prototype spec (iter-003, flow-interaction lens).

flow_divergence_60s: short-window (60s z) variant of the champion
flow_divergence_300s -- divergence horizon matched to 15-30s prediction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 20  # 20 x 3s rows = 60s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 60s) - z(trade_imbalance_60s, 60s)."""
    ofi = pl.col("ofi_60s")
    ti = pl.col("trade_imbalance_60s")
    return part.select((_z(ofi, W) - _z(ti, W)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="flow_divergence_60s",
    mechanism=(
        "Horizon-matched absorption/stealth divergence: the champion "
        "flow_divergence_300s compares z(ofi) and z(trade imbalance) over "
        "a 300s window -- appropriate for minute-scale absorption but "
        "over-smoothed for the 15-30s horizons where it also passes OOS. "
        "A 60s z window keeps the divergence FRESH: book building racing "
        "ahead of (or lagging) executed aggression within the last minute "
        "flags immediate stealth-limit activity whose price consequence is "
        "seconds away, not minutes. Same economic signal as the champion, "
        "bandwidth tuned to the short horizon."
    ),
    info_set="ofi_60s, trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 5: short-window variants of the "
        "flow_divergence_300s champion (passed 15/30/60s OOS on the "
        "2026-08-05 re-screen); CKS (2014) OFI vs signed volume; iceberg/"
        "stealth absorption (Buti & Rindi 2013)."
    ),
    compute=compute,
)
