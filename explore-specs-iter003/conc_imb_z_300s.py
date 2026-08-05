"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

conc_imb_z_300s: trailing-300s z-score of the depth-concentration
asymmetry -- the persistent placement-style regime (head-heavy vs
deep-heavy structure, bid vs ask).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(concentration asymmetry, 300s); warm-up rows null."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    cb = pl.when(tb > 0.0).then(db / tb).otherwise(pl.lit(None, dtype=pl.Float64))
    ca = pl.when(ta > 0.0).then(da / ta).otherwise(pl.lit(None, dtype=pl.Float64))
    return part.select(_z(cb - ca, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="conc_imb_z_300s",
    mechanism=(
        "Placement-style regime: the bid/ask concentration difference "
        "(top-5 share of each side's full-book volume) z-scored against "
        "its trailing-300s distribution. Persistent extremes mark a "
        "sustained strategic posture: a bid side parked at the head while "
        "the ask side stays deep is a durable market-making/inventory "
        "mode (e.g. creation/redemption programs working the visible "
        "queue on one side only), and such structural regimes decay slowly "
        "and condition how the next minutes trade -- head-heavy supported "
        "structure favors continuation of the favored side at 300-900s. "
        "The regime-z form separates persistent posture from the transient "
        "migration events measured by the momentum sibling, and keeps the "
        "construction in the live delta/z class rather than the dead "
        "level class."
    ),
    info_set="depth_bid5, depth_ask5, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R2-C family brief direction 3 (depth concentration "
        "regime; structural slow variables at 300s); book-shape "
        "literature (Zovio & Farmer 2002); slow-regime z convention per "
        "spread_z_300s built-in."
    ),
    compute=compute,
)
