"""Explore-lane prototype spec (iter-002, accumulation-regime lens).

trade_arrival_burst: Hawkes/ACD-style trade-arrival burst flag, built on the
materialized wishlist column ``n_trades_60s`` (first usable panel column for
this mechanism -- see library/candidates.json iter-001 needs_materialization).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

FAST = 100   # 100 x 3s rows = 300s fast arrival intensity
BASE = 600   # 600 x 3s rows = 1800s trailing baseline window


def compute(part: pl.DataFrame) -> pl.Series:
    """Z-score of the 300s arrival intensity vs its trailing-1800s baseline.

    n_trades_60s is the engine's trailing-60s trade count (null warm-up).
    The fast intensity smooths it to a 300s rate; the baseline mean/std are
    computed over the trailing 1800s of that fast intensity, so a burst is
    scored against the recent regime, not the whole day. Constant baselines
    (std == 0) map to 0.0 (neutral) per the causal_zscore convention; rows
    inside the 1800s warm-up are null, never zero-filled.
    """
    n = pl.col("n_trades_60s")
    fast = n.rolling_mean(window_size=FAST, min_samples=FAST)
    base_mean = fast.rolling_mean(window_size=BASE, min_samples=BASE)
    base_std = fast.rolling_std(window_size=BASE, min_samples=BASE)
    z = (fast - base_mean) / base_std
    return part.select(
        pl.when(base_std.is_not_null() & (base_std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="trade_arrival_burst",
    mechanism=(
        "Trade-arrival bursts (Hawkes self-excitation / ACD): information "
        "events make trading cluster in time, so a burst in the arrival rate "
        "relative to its trailing baseline flags the START of an information "
        "episode that then unfolds over minutes. Unlike ofi/trade_imbalance "
        "(signed direction of flow) this is direction-free INTENSITY: a burst "
        "says 'something is happening', and on 300/900s horizons the "
        "continuation of the episode (in both volatility and drift of the "
        "already-revealed direction) should be forecastable. Complements the "
        "size-blind flow sums with a timing signal; iter-001 established the "
        "mechanism was untestable without trade counts -- n_trades_60s was "
        "materialized exactly to unblock it."
    ),
    info_set="n_trades_60s (wishlist, materialized 2026-08-05)",
    inspiration=(
        "Engle & Russell (1998) ACD; Hawkes (1971) self-excitation; iter-001 "
        "candidates.json needs_materialization entry (accumulation-regime "
        "lens); digest: arrival dynamics only testable via snapshot-delta "
        "reconstruction on SSE -- now available as an engine column."
    ),
    compute=compute,
)
