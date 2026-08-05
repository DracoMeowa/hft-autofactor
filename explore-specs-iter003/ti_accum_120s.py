"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ti_accum_120s: SHORT-window (120s) signed aggressive-flow accumulation --
deliberately away from the crowded 300s ti accumulator cluster.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing accumulation window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 120s mean of trade_imbalance_60s; warm-up rows null."""
    x = pl.col("trade_imbalance_60s")
    return part.select(
        x.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ti_accum_120s",
    mechanism=(
        "Short-window signed-flow accumulation: integrated aggressive "
        "imbalance over the last two minutes tracks meta-order progress "
        "on a timescale matched to the 15-60s prediction horizons, where "
        "the registered 300s accumulators (ti_accum_300s / ti_ewm_state "
        "/ dd_flow, mutually 0.86-0.95 correlated) are already crowded "
        "and over-smoothed. A 120s accumulator reacts to fresh one-sided "
        "aggression within ~30s of its start (the 60s engine window "
        "rolling into the sum), keeping the pressure estimate fresh: "
        "sustained net buying over two minutes predicts continuation at "
        "the next 15-60s; recently flipped accumulation predicts the "
        "turn. Short window + same economic object, different bandwidth."
    ),
    info_set="trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 12: short-window accumulator variant "
        "explicitly positioned away from the 300s ti cluster flagged in "
        "the archive lessons; meta-order pressure persistence (Bouchaud, "
        "Gefen, Potters & Wyart 2004)."
    ),
    compute=compute,
)
