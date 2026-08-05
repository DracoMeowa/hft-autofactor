"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

ti_pos_frac_300s: fraction of the trailing 100 rows (300s) where
trade_imbalance_60s > 0 -- the DURATION of one-sided aggressive-flow
regimes, not their intensity.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing persistence window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing fraction of rows with trade_imbalance_60s > 0.

    Null inputs stay null (so engine warm-up gaps are excluded from the
    fraction); warm-up (< 100 non-null rows) is null, never zero-filled.
    """
    x = pl.col("trade_imbalance_60s")
    ind = (
        pl.when(x.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(x > 0)
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select(
        ind.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ti_pos_frac_300s",
    mechanism=(
        "Magnitude accumulators (ti_accum_300s and kin) conflate two very "
        "different regimes: 30 snapshots of strong one-sided buying and "
        "95 snapshots of weak-but-persistent buying can average to the "
        "same value, yet the second is the signature of a patient meta-"
        "order being worked in small slices (Kyle 1985: informed traders "
        "hide by spreading over time) while the first is more often a "
        "burst that exhausts. The trailing-300s FRACTION of snapshots "
        "with positive trade imbalance measures direction DURATION: "
        "values near 1 mean buying dominated almost every 3s snapshot for "
        "five minutes -- a persistent regime that continues at 300-900s "
        "because the schedule driving it is not finished; values near 0.5 "
        "mean a balanced/churn tape with no regime. The sign-frequency "
        "statistic is nearly orthogonal to any magnitude accumulator by "
        "construction (it ignores |TI| entirely), opening the persistence "
        "dimension the crowded ti cluster never touched."
    ),
    info_set="trade_imbalance_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 3 (regime persistence: "
        "share of trailing rows with positive flow). Meta-order slicing "
        "persistence (Kyle 1985; Bouchaud et al. 2004); deliberately "
        "orthogonal statistic vs the registered ti accumulator cluster."
    ),
    compute=compute,
)
