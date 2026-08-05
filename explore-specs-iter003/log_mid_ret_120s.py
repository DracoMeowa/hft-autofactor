"""Explore-lane prototype spec (iter-003, price-vol family).

log_mid_ret_120s: slow signed mid-price momentum (120s = 40 x 3s rows).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 40 rows x 3s = 120s trailing window
K = 40


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing-120s log-mid return; warm-up rows null (diff semantics)."""
    return part.select(pl.col("mid_px").log().diff(K).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="log_mid_ret_120s",
    mechanism=(
        "Two-minute signed mid momentum: bridges the fast 60s drift of the "
        "built-in and the slower accumulation state. A sustained 120s "
        "directional move reflects a multi-minute order-flow program (index "
        "rebalance, creation/redemption, informed accumulation) rather than "
        "a single burst, and such programs tend to persist (continuation "
        "into 60-300s) until their target size is filled. Conversely, a "
        "120s move that has run far relative to recent vol flags overshoot "
        "and reversal. The 120s window decorrelates from the 60s window "
        "while still being fast enough for the short-horizon gate."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 price-vol family brief: 'momentum exists in the built-in "
        "log_mid_ret_60s family -- vary windows'. Slow-momentum companion "
        "of log_mid_ret_60s; order-program persistence (Vayanos & Wang 2013, "
        "flow persistence)."
    ),
    compute=compute,
)
