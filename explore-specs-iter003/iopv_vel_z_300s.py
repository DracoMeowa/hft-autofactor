"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

iopv_vel_z_300s: causal z-score of iopv_velocity (IOPV 60s change rate,
bps/s) against its trailing 300s distribution.  Fast fundamental
re-pricing unusual vs recent minutes.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing reference window


def compute(part: pl.DataFrame) -> pl.Series:
    """Causal z-score of iopv_velocity over 100 rows.

    Warm-up (< 100 rows) null; constant trailing windows (std == 0) map to
    0.0 (neutral) per the causal_zscore convention.
    """
    x = pl.col("iopv_velocity")
    mean = x.rolling_mean(window_size=W, min_samples=W)
    std = x.rolling_std(window_size=W, min_samples=W)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="iopv_vel_z_300s",
    mechanism=(
        "IOPV is the arbitrage anchor of an ETF: when the NAV benchmark "
        "itself starts moving unusually fast versus its own trailing-300s "
        "behavior, a fundamental re-pricing of the underlying basket is "
        "underway. The ETF mid follows IOPV with a lag because arbitrage "
        "execution (creation/redemption, basket hedging) takes seconds to "
        "minutes, so a velocity spike marks the ONSET of a fair-value move "
        "the ETF price has not yet fully tracked: the current velocity, "
        "when unusually large, predicts continuation of the ETF mid in the "
        "velocity direction at 60-900s as arb flow closes the tracking "
        "gap. This is NOT a premium-level factor (all 8 unconditional "
        "premium/IOPV forms died of regime break): the input is the rate "
        "of change of the anchor (batch-2 column iopv_velocity), and the "
        "transform is regime-relative (z vs trailing 300s), so it measures "
        "fresh arbitrage-pressure ACCELERATION rather than the dead "
        "level/momentum of the premium gap."
    ),
    info_set="iopv_velocity",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 1 (iopv_velocity rolling "
        "z): velocity/accumulation angles are the only surviving lane for "
        "the ETF-arbitrage family after the round-1 0/8 wipeout of "
        "unconditional premium levels. ETF price-IOPV lead-lag tracking "
        "via creation/redemption latency."
    ),
    compute=compute,
)
