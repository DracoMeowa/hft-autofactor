"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

ntrades_pace_z_300s: z-score of the 300s-smoothed trade ARRIVAL rate
(n_trades_60s) -- a trade-pacing regime detector.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s smoothing AND z reference window


def compute(part: pl.DataFrame) -> pl.Series:
    """z of rolling_mean(n_trades_60s, 100 rows) vs trailing 100 rows.

    Warm-up (~199 rows) null; constant trailing windows (std == 0) map to
    0.0 (neutral).
    """
    x = pl.col("n_trades_60s").rolling_mean(window_size=W, min_samples=W)
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
    name="ntrades_pace_z_300s",
    mechanism=(
        "The trade arrival rate, smoothed over five minutes and judged "
        "against its own trailing-300s norm, measures the participation "
        "REGIME: unusually dense trading persists because order flow is "
        "self-exciting (Hawkes clustering runs for minutes) and because "
        "execution algorithms ramp gradually. On this instrument the "
        "round-1 evidence shows the contrarian edge lives in overheated "
        "states: near-day-high rejection, negative 900s IC of "
        "log_mid_ret_120s and signed_rv_60s. Overheated participation is "
        "where that heat is generated -- chase flows cluster in dense "
        "tapes and get absorbed -- so an unusually high pacing regime "
        "predicts reversion (negative IC) at 300-900s, while unusually "
        "sparse tape marks quiet drift with nothing to fade. Smoothing "
        "first, then z-scoring, removes both the 3s snapshot noise and "
        "the U-shaped time-of-day seasonality of raw trade counts."
    ),
    info_set="n_trades_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 6 (trade-pacing regime: "
        "300s rolling_mean of n_trades_60s, then z). Hawkes persistence "
        "(Bacry et al. 2015); round-1 contrarian evidence on overheated "
        "states of 588000 at long horizons."
    ),
    compute=compute,
)
