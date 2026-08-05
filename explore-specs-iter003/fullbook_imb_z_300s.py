"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

fullbook_imb_z_300s: trailing-300s z-score of the WHOLE-book bid/ask
volume imbalance -- the slow structural regime of broad-book pressure.
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
    """z(full-book imbalance, 300s); warm-up rows null."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(_z(fbi, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_z_300s",
    mechanism=(
        "Broad-book pressure regime: the whole-book bid/ask imbalance "
        "z-scored against its own trailing-300s distribution. The outer "
        "queue aggregated by total_bid_vol/total_ask_vol is where patient "
        "institutional and ETF creation/redemption positioning parks, so "
        "the full-book ratio moves more slowly than top-of-book measures "
        "and its z-extremes mark sustained commitment regimes: an unusually "
        "bid-skewed whole book is a reservoir that keeps refilling the "
        "touch, supporting continued one-sided drift at 300-900s; the "
        "mirror flags sustained distribution. Z-scoring converts the "
        "non-stationary imbalance level into a regime state -- the "
        "structural slow-variable member of the full-book pair, aimed at "
        "the horizons where accumulation/regime state paid off in the "
        "eval-v2 re-screen."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R2-C family brief direction 1 (full-book imbalance and "
        "its z); slow-regime z convention of the spread_z_300s built-in; "
        "meta-lesson: level statistics dead, regime-z/deltas live; "
        "eval-v2 re-screen: 4/6 passes were at the 900s horizon."
    ),
    compute=compute,
)
