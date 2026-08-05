"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

netbuy_pressure_delta_60s: 60s CHANGE of the side-attributed net buying
pressure built from the new buy_vol_60s / sell_vol_60s columns.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

LAG = 20  # 20 x 3s rows = 60s change horizon


def _netbuy_pressure() -> pl.Expr:
    """(buy - sell)/(buy + sell); null when total aggressive volume is 0."""
    buy = pl.col("buy_vol_60s")
    sell = pl.col("sell_vol_60s")
    tot = buy + sell
    return (
        pl.when(tot.is_not_null() & (tot > 0.0))
        .then((buy - sell) / tot)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """netbuy_pressure.diff(60s); warm-up rows null."""
    return part.select(_netbuy_pressure().diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="netbuy_pressure_delta_60s",
    mechanism=(
        "Buildup/exhaustion of aggressive buying: the side-attributed net "
        "buy pressure (buy_vol - sell_vol)/(buy_vol + sell_vol) measures "
        "what fraction of the last minute's aggressive volume was net "
        "buying; its 60s change isolates whether that pressure is BUILDING "
        "or FADING right now, independent of its level. A market can sit "
        "at moderately positive pressure for minutes while the information "
        "is already in the price; the rising edge is the entry of a fresh "
        "one-sided participant and is what precedes 15-60s continuation. "
        "Falling pressure from a high base flags exhaustion ahead of "
        "stalls. Delta form per the round-1 meta-lesson (levels of state "
        "die, changes carry signal). Constructed from the new batch-2 "
        "side-attributed volumes, whose attribution/normalization differs "
        "from the engine's trade_imbalance_60s; correlation with that "
        "panel column must stay below the 0.85 gate and the diff form is "
        "the primary decorrelation lever."
    ),
    info_set="buy_vol_60s, sell_vol_60s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 3 (net buy/sell pressure and its "
        "diff); buy_vol_60s/sell_vol_60s materialized 2026-08-06; "
        "order-flow change-over-level meta-lesson from round 1."
    ),
    compute=compute,
)
