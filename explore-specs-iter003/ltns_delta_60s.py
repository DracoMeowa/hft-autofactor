"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ltns_delta_60s: 60s CHANGE of the signed large-trade net share -- are the
whales turning or intensifying?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

LAG = 20  # 20 x 3s rows = 60s change horizon


def compute(part: pl.DataFrame) -> pl.Series:
    """large_trade_net_share_60s.diff(60s); warm-up rows null."""
    return part.select(
        pl.col("large_trade_net_share_60s").diff(LAG).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ltns_delta_60s",
    mechanism=(
        "Turning of the whale footprint: the 60s change of the signed "
        "large-trade net share measures whether the biggest players flipped "
        "or intensified their direction over the last minute. The LEVEL of "
        "large-order net direction can persist for long stretches while an "
        "institutional program executes (already in the price); the 60s "
        "change isolates the fresh component -- the moment large flow "
        "starts, stops, or reverses. A turn toward positive means big "
        "tickets just became net buyers versus a minute ago, predicting "
        "15-60s continuation up; a turn negative flags distribution "
        "starting. Delta form per the round-1 change-over-level lesson. "
        "Orthogonal axis to ltns_x_ti_60s (which conditions on concurrent "
        "aggression): this captures large-flow TIMING, not confirmation."
    ),
    info_set="large_trade_net_share_60s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 4 (diff of signed large-trade net "
        "share); large_trade_net_share_60s materialized 2026-08-06; "
        "change-over-level meta-lesson from round 1."
    ),
    compute=compute,
)
