"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

size_regime_z_300s: z-score of the 300s-smoothed average per-trade SIZE
(avg_trade_size_60s) -- the composition regime of participation.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s smoothing AND z reference window


def compute(part: pl.DataFrame) -> pl.Series:
    """z of rolling_mean(avg_trade_size_60s, 100 rows) vs trailing 100 rows.

    Warm-up (~199 rows) null; constant trailing windows (std == 0) map to
    0.0 (neutral).
    """
    x = pl.col("avg_trade_size_60s").rolling_mean(window_size=W, min_samples=W)
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
    name="size_regime_z_300s",
    mechanism=(
        "Average per-trade size, smoothed over five minutes and compared "
        "to its own trailing norm, measures WHO is trading: unusually "
        "large average prints mean the tape is dominated by "
        "institutional/algorithmic parents (execution algos print in "
        "scheduled chunks; retail flow stays small), while unusually "
        "small prints mark retail churn. Institutional participation "
        "regimes persist by construction of the schedules driving them "
        "(parents run for many minutes), and institutional flow is the "
        "informed flow whose price impact decays slowly (square-root "
        "law): an unusually large-size regime predicts that the "
        "concurrent drift -- whatever direction the tape is leaning -- "
        "continues, which as a symmetric regressor surfaces as reversion "
        "of the retail-churn episodes that dominate the complementary "
        "state (negative IC on the churn side at 300-900s, consistent "
        "with round-1's contrarian long-horizon evidence). Distinct from "
        "the dead large_share_mom_300s: different input (mean size, not "
        "top-10% volume share) and different transform (regime z, not "
        "level momentum)."
    ),
    info_set="avg_trade_size_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 6 (300s rolling_mean of "
        "avg_trade_size_60s, then z). Trade-size information content "
        "(Barclay & Warner 1995); meta-order schedule persistence; "
        "round-1 contrarian evidence at long horizons."
    ),
    compute=compute,
)
