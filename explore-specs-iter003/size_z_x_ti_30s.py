"""Explore-lane prototype spec (iter-003 R2-C, trade-structure lens).

size_z_x_ti_30s: granularity regime surge gated by the FAST 30s
aggressive trade imbalance (magnitude-weighted) -- big tickets with
fresh one-sided aggression.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window for trade size


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(log avg_trade_size_60s, 300s) x trade_imbalance_30s; warm-up null."""
    size = pl.col("avg_trade_size_60s")
    x = (
        pl.when(size > 0.0)
        .then(size.log())
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    size_z = _z(x, W)
    ti = pl.col("trade_imbalance_30s")
    val = size_z * ti
    return part.select(
        pl.when(size_z.is_not_null() & ti.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="size_z_x_ti_30s",
    mechanism=(
        "Directional confirmation of the size regime: big tickets alone "
        "are directionless, so the granularity surge (z of log average "
        "trade size vs its trailing-300s regime) is multiplied by the "
        "concurrent FAST 30s aggressive trade imbalance, which supplies "
        "fresh aggressor direction AND conviction magnitude. Large prints "
        "arriving together with net one-sided aggression are informed/"
        "institutional execution on a specific side (Kyle: informed "
        "traders trade large AND directionally), and meta-order execution "
        "persists, so the signed direction continues at 15-30s; size "
        "surges with balanced aggression score near zero by construction "
        "(two-sided churn). Deliberate refinements over round-1's "
        "size_x_direction: magnitude-weighted 30s direction instead of "
        "sign-only 60s, and a 300s size regime window instead of 180s."
    ),
    info_set="avg_trade_size_60s, trade_imbalance_30s (batch-2)",
    inspiration=(
        "iter-003 R2-C family brief direction 6 (large-trade directional "
        "confirmation); Kyle (1985) informed flow is large and "
        "directional; refinement of round-1 size_x_direction using the "
        "batch-2 fast trade_imbalance_30s channel."
    ),
    compute=compute,
)
