"""Explore-lane prototype spec (iter-003 R5, family R5-C).

wdi_zcross_x_rvlo: depth-imbalance regime-flip velocity ISOLATED to the
low-volatility (calm) regime (rv_60s at or below its 300s rolling mean).
Tests whether depth-stack rebuilds carry long-horizon signal only when
the market is quiet.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z / regime window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """wdi crossing velocity, kept only when rv <= rv_mean; else 0. warm-up null."""
    z = _z(pl.col("wdi"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    cross_vel = flip * (z - z_lag)
    rv = pl.col("rv_60s")
    rv_mean = rv.rolling_mean(window_size=W, min_samples=W)
    gate = (
        pl.when(rv_mean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(rv > rv_mean)
        .then(pl.lit(0.0))
        .otherwise(pl.lit(1.0))
    )
    return part.select((cross_vel * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zcross_x_rvlo",
    mechanism=(
        "Calm-isolated depth-stack rebuilds: wdi_z_cross_vel_15s scores "
        "sign-reversal events where the entire visible 5-level bid/ask "
        "stack rebuilds against a minutes-old tilt. Hypothesis: in CALM "
        "regimes (rv at or below its trailing mean) the book trades off "
        "queue structure and patient orders get filled in order, so a "
        "deliberate full-stack rebuild is credible as informed "
        "repositioning whose direction continues at 300-900s. In "
        "TURBULENCE, exogenous shocks run over queue positioning and the "
        "same rebuild is reactive panic, not information -- the signal "
        "decays. The binary 0/1 gate zeroes all turbulent-regime rebuilds "
        "and retains only calm-regime ones. This is the OPPOSING "
        "falsifiable hypothesis to oir_zcross_x_rvhi: that one says "
        "turbulence is the necessary condition (urgency); this one says "
        "calm is (deliberate informed positioning). On a different base "
        "(wdi 5-level crossing vs oir touch crossing) and different regime "
        "polarity, so the pair cleanly spans the two economic answers to "
        "'when does book-structure regime change carry signal'."
    ),
    info_set="wdi, rv_60s",
    inspiration=(
        "iter-003 R5-C family brief direction 2: condition z-vel winners "
        "on rv regime; this spec isolates the low-rv (calm) regime only "
        "(quiet-market hypothesis, competing with oir_zcross_x_rvhi) on "
        "the wdi crossing base."
    ),
    compute=compute,
)
