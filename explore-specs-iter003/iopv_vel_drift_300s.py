"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

iopv_vel_drift_300s: trailing 300s mean of iopv_velocity -- the persistence
of the IOPV drift direction (sustained arbitrage-pressure trend, not a
snapshot).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing accumulation window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 300s mean of iopv_velocity; warm-up rows null."""
    x = pl.col("iopv_velocity")
    return part.select(
        x.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="iopv_vel_drift_300s",
    mechanism=(
        "Averaging the 60s IOPV change rate over five minutes yields the "
        "SUSTAINED drift of the arbitrage anchor: whether NAV has been "
        "persistently rising or falling, not just flickering. One-off "
        "velocity spikes can be index recomputation noise or single-stock "
        "shocks that fade; a 300s-persistent velocity is a genuine "
        "fundamental trend regime in the underlying basket. Under such a "
        "regime, arbitrage flow works the ETF toward the moving anchor "
        "over many minutes, so a positive sustained drift predicts "
        "continued ETF appreciation at the 300-900s horizons (and mirror). "
        "This is the direction-PERSISTENCE companion of iopv_vel_z_300s: "
        "the z asks 'is velocity unusual now', the drift asks 'has the "
        "anchor been trending'; the two decorrelate during decaying "
        "episodes (z normalizes, drift stays signed). Both avoid the dead "
        "premium-level lane: the input is NAV velocity, a batch-2 column."
    ),
    info_set="iopv_velocity",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 1 (rolling_mean of "
        "iopv_velocity = persistence of premium-drift direction). Slow "
        "regime-state construction per the round-1 meta-lesson (300-900s "
        "horizons reward accumulation/regime states, 4/6 of eval-v2 "
        "re-screen passes at 900s)."
    ),
    compute=compute,
)
