"""Explore-lane prototype spec (iter-003 R5-D, spread thin-gate fill-in).

ofi_zvel_thin_gate: OFI regime-crossing events (ofi_z_cross_vel_15s) active
ONLY when the raw spread is below its rolling mean -- book-flow regime
switches in CHEAP-to-trade regimes where the arbitrage is easiest.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window on ofi_60s and spread mean
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """ofi_z_cross_vel_15s base x 1{spread < rolling_mean}; warm-up null.

    The base is: (z_now - z_15s_ago) where sign(z) flipped over 15s,
    else 0. The gate zeros out crossing events that land in the wide-
    spread subset, keeping only thin-spread crossings.
    """
    z = _z(pl.col("ofi_60s"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    base = flip * (z - z_lag)
    sp = pl.col("quoted_spread_ticks").cast(pl.Float64)
    sp_mean = sp.rolling_mean(window_size=W, min_samples=W)
    gate = (
        pl.when(sp_mean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(sp < sp_mean)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
    )
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_zvel_thin_gate",
    mechanism=(
        "Regime-crossing events active ONLY under THIN quoting: the "
        "ofi_z_cross_vel_15s base (velocity of the moment the 300s z of "
        "ofi_60s crosses zero within 15s) is zeroed UNLESS the raw "
        "quoted spread is below its trailing mean -- the comfortable, "
        "cheap-to-trade subset. The economic claim is the OPPOSITE of "
        "the wide-gate family: a book-flow regime switch is MORE "
        "informative when the spread is thin, because the arbitrage that "
        "closes the gap is cheapest to execute there -- informed "
        "participants can act on the crossing at minimum cost, so the "
        "new direction asserts itself fastest in the tight regime. "
        "Under wide spreads the same crossing faces an expensive toll "
        "and is slower to resolve; this spec is switched off there with "
        "no continuation claim. This is the first thin-gate spec in the "
        "library (all prior spread interactions gate the wide subset); "
        "the binary on/off in the cheap regime is a genuinely different "
        "economic question from any wide-gate sibling. Dedup note: the "
        "nonzero support is a strict subset of ofi_z_cross_vel_15s "
        "(round-3 admitted), so panel corr may be moderate, but the "
        "regime selection (thin half only) is the distinct input."
    ),
    info_set="ofi_60s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-D brief direction (a): thin-spread gate variant "
        "for the ofi_z_cross_vel_15s base (round-3 admitted); tests "
        "whether the crossing signal is sharper in the cheap-arb "
        "regime, opposite of the wide-gate stress hypothesis."
    ),
    compute=compute,
)
