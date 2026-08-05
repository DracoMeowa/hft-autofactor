"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ofi_regime_mature_60s: book-flow regime MATURITY -- the 300s z of ofi_60s
has kept its sign unbroken for >= 60s; value = the current regime z, else 0.
Continuation weight on mature regimes only.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_Z = 100  # 300s trailing z window on ofi_60s
K = 20     # 20 consecutive same-sign transitions = >= 63s unbroken regime


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z_ofi_now if z-sign held unbroken over the last 20 transitions, else 0.

    rolling_min over the same-sign transition indicators is 1.0 iff every
    one of the last K transitions kept the sign (regime mature); warm-up
    rows null, never zero-filled.
    """
    z = _z(pl.col("ofi_60s"), W_Z)
    s = z.sign()
    s_lag = s.shift(1)
    same = (
        pl.when(s.is_null() | s_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((s == s_lag) & (s != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    mature = same.rolling_min(window_size=K, min_samples=K)
    return part.select((mature * z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_regime_mature_60s",
    mechanism=(
        "Mature flow regimes continue, fresh ones are unproven: the "
        "trailing-300s z of ofi_60s keeping its sign unbroken for at "
        "least 60s means the book's flow regime has survived more than "
        "twenty consecutive 3s snapshots without a single reversal -- a "
        "committed, persistent reallocation of limit interest rather "
        "than noise. Persistence is the documented amplifier of OFI's "
        "price impact (CKS 2014), so the factor passes the current "
        "regime z through only when mature and scores young/just-flipped "
        "regimes as 0 -- the continuation bet is placed only on regimes "
        "with demonstrated staying power. This partitions the regime-"
        "switch family cleanly: ofi_z_cross_vel_15s owns the fresh "
        "crossing events, ofi_flip_fade_300s owns unstable chop, and "
        "this factor owns the mature regime cell. The statistic is a "
        "duration gate, not a level: it cannot reduce to ofi_60s or its "
        "z alone because identical z values score differently depending "
        "on how long the sign has held."
    ),
    info_set="ofi_60s (library)",
    inspiration=(
        "iter-003 R3-C brief direction 5 (OFI regime switches: causal "
        "rolling comparison of current vs lagged regime state); CKS "
        "(2014) persistence-monotone impact; the mature-regime cell of "
        "the cross/fade/maturity partition."
    ),
    compute=compute,
)
