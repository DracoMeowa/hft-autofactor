"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

top5_book_div_z_300s: trailing-300s z-score of the divergence between
the 5-level imbalance (wdi) and the full-book imbalance -- sustained
touch-vs-queue structural mismatch regime.
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
    """z(wdi - full-book imbalance, 300s); warm-up rows null."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(_z(pl.col("wdi") - fbi, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="top5_book_div_z_300s",
    mechanism=(
        "Structural mismatch regime: the (wdi minus full-book imbalance) "
        "gap z-scored against its trailing-300s distribution. A "
        "persistently POSITIVE z means displayed touch strength is NOT "
        "backed by the deep queue -- thin-backed strength that is fragile: "
        "the touch gets consumed without deep refill, hypothesized to "
        "drift DOWN over 300-900s (negative contribution). A persistently "
        "NEGATIVE z means a deep bid reservoir with modest visible touch "
        "-- a supported floor where strength is queued but not yet "
        "displayed. When the two imbalances have opposite signs the gap "
        "reaches its maximum (|wdi| + |fbi|), so the same statistic also "
        "catches sign-level structural conflict. Regime-z form separates "
        "sustained mismatch from the transient flickers measured by the "
        "momentum sibling."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R2-C family brief direction 4 (five-level vs full-book "
        "divergence as a structural slow variable at 300s); thin-backed "
        "displayed strength / spoof-like fragility hypothesis; slow-"
        "regime z convention per spread_z_300s built-in."
    ),
    compute=compute,
)
