"""Explore-lane prototype spec (iter-001, flow-queue lens).

ti_ewm_accel_120s: aggressive-flow acceleration (fast-minus-slow EWMA).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

ALPHA_FAST = 1.0 - 0.5 ** (1.0 / 5.0)    # half-life 15s  = 5 rows
ALPHA_SLOW = 1.0 - 0.5 ** (1.0 / 20.0)   # half-life 60s  = 20 rows
MIN_FAST, MIN_SLOW = 20, 60


def compute(part: pl.DataFrame) -> pl.Series:
    x = pl.col("trade_imbalance_60s")
    fast = x.ewm_mean(alpha=ALPHA_FAST, adjust=False, min_samples=MIN_FAST,
                      ignore_nulls=True)
    slow = x.ewm_mean(alpha=ALPHA_SLOW, adjust=False, min_samples=MIN_SLOW,
                      ignore_nulls=True)
    return part.select((fast - slow).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_ewm_accel_120s",
    mechanism=(
        "Aggressive-flow acceleration: fast EWMA (15s half-life) minus slow "
        "EWMA (60s half-life) of signed trade imbalance - a MACD of "
        "aggressor flow. The LEVEL of imbalance says where flow has been; "
        "the acceleration says where it is turning. A burst of one-sided "
        "aggression (sweeping several book levels within seconds) precedes "
        "short-horizon continuation (queue depletion cascades), while a "
        "deceleration of previously strong flow precedes stalls/reversion. "
        "Because it is a difference of two backward filters of the same "
        "series it is nearly orthogonal to the flow LEVEL factors by "
        "construction."
    ),
    info_set="trade_imbalance_60s (library)",
    inspiration=(
        "Digest iter-000: flow signals die by 900s and peak at 15s - the "
        "fresh information in flow is its CHANGE, not its level; "
        "momentum-trigger logic in order-flow bursts (Cartea-Jaimungal-"
        "Penalva 2015, ch.8 aggressive-order dynamics)."
    ),
    compute=compute,
)
