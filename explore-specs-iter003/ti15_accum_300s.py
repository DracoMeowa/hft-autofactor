"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

ti15_accum_300s: trailing 300s (100 x 3s rows) mean of the FAST 15s trade
imbalance -- aggressive-flow accumulation at 4x the update rate of the
registered 60s-based ti_accum_300s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing accumulation window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 300s mean of trade_imbalance_15s; warm-up rows null."""
    x = pl.col("trade_imbalance_15s")
    return part.select(
        x.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ti15_accum_300s",
    mechanism=(
        "Informed meta-orders are worked over minutes, so the integrated "
        "signed aggressive imbalance is the natural slow state -- but the "
        "registered ti_accum_300s averages OVERLAPPING 60s windows, whose "
        "effective information bandwidth is only ~1/4 of the nominal "
        "window and reacts sluggishly to fresh one-sided aggression. "
        "Averaging 100 rows of the batch-2 15s imbalance instead keeps a "
        "300s lookback while sampling the aggressor flow at 4x the rate: "
        "the accumulator turns within ~15-30s of a regime onset and "
        "resolves burst structure the 60s window smooths away. Sustained "
        "positive accumulation = persistent net buying aggression still "
        "being worked -> continuation at 300-900s; the extra bandwidth "
        "should decorrelate from the 60s-based sibling exactly during "
        "onset/flip episodes, which is where the predictive content of "
        "accumulation lives (Bouchaud et al. 2004 impact power law, "
        "G(t) ~ t^-1/2: fresh flow keeps moving price for minutes)."
    ),
    info_set="trade_imbalance_15s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 2 (300s accumulation of "
        "the new fast flow columns). Deliberately distinct from the "
        "registered ti_accum_300s/ti_ewm_state_300s/dd_flow_300s cluster "
        "(mutual rho 0.86-0.96): same economic object, finer sampling "
        "bandwidth via the batch-2 trade_imbalance_15s column."
    ),
    compute=compute,
)
