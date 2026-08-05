"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ofi_fast_slow: fast-minus-slow trailing-mean crossover of BOOK flow (OFI) --
the seed's ti_fast_slow regime-shift idea moved to the uncrowded passive
channel (the trade channel already owns ti_ewm_accel_120s and the 300s ti
accumulator cluster).

Implementation note: the exponential (ewm_mean) version was abandoned
during smoke testing -- polars ewm_mean contaminates its state to NaN when
the input column has LEADING nulls (as ofi_60s does during its 60s engine
warm-up) and never recovers, in both ignore_nulls modes. Fixed-window
rolling means give the same fast/slow crossover semantics and are
null-robust and strictly causal.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_FAST = 10  # 10 x 3s rows = 30s trailing fast state
W_SLOW = 60  # 60 x 3s rows = 180s trailing slow state


def compute(part: pl.DataFrame) -> pl.Series:
    """rolling_mean(ofi_60s, 30s) - rolling_mean(ofi_60s, 180s); null warm-up."""
    x = pl.col("ofi_60s")
    fast = x.rolling_mean(window_size=W_FAST, min_samples=W_FAST)
    slow = x.rolling_mean(window_size=W_SLOW, min_samples=W_SLOW)
    return part.select((fast - slow).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_fast_slow",
    mechanism=(
        "Book-flow regime shift: the fast-minus-slow trailing-mean "
        "crossover of OFI flags the moment limit-flow turns. The fast leg "
        "(30s mean) reacts within seconds of a change in book-building "
        "behavior; while it stays above the 180s slow leg, the passive "
        "side is in a fresh accumulating regime. The crossover is a "
        "REGIME-SHIFT statistic, nearly orthogonal to any flow LEVEL: a "
        "cross from below to above predicts up-continuation at 15-60s as "
        "freshly revealed book-building demand works through the queue; "
        "the mirror cross flags distribution. OFI channel chosen "
        "deliberately: the trade-imbalance channel already has a "
        "registered fast-slow crossover (ti_ewm_accel_120s) and a "
        "crowded 300s accumulator cluster, while no fast/slow state "
        "exists for book flow."
    ),
    info_set="ofi_60s (library)",
    inspiration=(
        "iter-003 family brief seed 14 (flow regime shift via fast-minus-"
        "slow smoothing), refined: the trade channel is crowded "
        "(ti_ewm_accel_120s registered; ti_ewm_state/ti_accum/dd_flow "
        "cluster 0.86-0.95), so the crossover is applied to the OFI "
        "channel; fixed windows replace EWMA kernels for null-robustness "
        "against engine warm-up gaps; MACD-style logic (Cartea-"
        "Jaimungal-Penalva 2015)."
    ),
    compute=compute,
)
