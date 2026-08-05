"""Explore-lane prototype spec (iter-003, etf-regime lens).

iopv_premium_z_120s: IOPV premium relative to its own trailing-120s
distribution.  Neither the raw level-momentum (iopv_premium_mom, dead
iter-001) nor the z(prem,100)xz(flow,100) interactions (prem_x_ofi /
prem_x_wdi, dead iter-001): a univariate relative-stretch transform with a
FAST reference frame matched to arbitrage response latency.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 120s (40 x 3s rows) causal z-score window
Z_WINDOW = 40


def compute(part: pl.DataFrame) -> pl.Series:
    """Causal z-score of iopv_premium over 40 rows.

    Warm-up (< 40 rows) is null; constant trailing windows (std == 0) map
    to 0.0 (neutral) per the causal_zscore convention.
    """
    x = pl.col("iopv_premium")
    mean = x.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = x.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="iopv_premium_z_120s",
    mechanism=(
        "Relative mispricing, not absolute: the ETF premium vs IOPV has a "
        "structural component (fund-flow regimes that persist for hours and "
        "carry no short-horizon information) and a transient component "
        "(flow bursts, stale IOPV after fast basket moves). Arbitrageurs "
        "react to the transient part -- premium away from where it has been "
        "in the last couple of minutes -- because creation/redemption "
        "economics are defined relative to the prevailing premium regime. "
        "Z-scoring over trailing 120s strips the structural drift and "
        "isolates stretch episodes: a premium 2 sigmas above its recent "
        "range means the ETF just ran ahead of fair value faster than "
        "arbitrage flow has responded, so reversion pressure toward IOPV is "
        "maximal now (high values predict negative forward returns). The "
        "dead iter-001 forms tested raw level-momentum and flow-"
        "interactions of the premium; the regime-adaptive univariate "
        "stretch is the untested object."
    ),
    info_set="iopv_premium",
    inspiration=(
        "iter-001/iter-002 archive lessons: iopv_premium_mom, prem_x_ofi, "
        "prem_x_wdi all IC ~ 0 -- raw level/momentum and z x flow products "
        "of the premium are dead; iter-003 etf-regime brief: premium "
        "RELATIVE to its own day/history is the open lane. AP "
        "creation/redemption arbitrage economics; limits to arbitrage "
        "(Shleifer & Vishny 1997) motivate conditioning on the recent "
        "regime rather than absolute levels."
    ),
    compute=compute,
)
