"""Explore-lane prototype spec (iter-003, depth-book lens).

top_book_delta_120s: 120s delta of the TOP-of-book quantity imbalance
-- sustained directional touch-queue accumulation trend.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 40  # 40 x 3s rows = 120s delta


def compute(part: pl.DataFrame) -> pl.Series:
    bq = pl.col("bid1_qty").cast(pl.Float64)
    aq = pl.col("ask1_qty").cast(pl.Float64)
    tot = bq + aq
    imb = (
        pl.when(tot > 0.0)
        .then((bq - aq) / tot)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(imb.diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="top_book_delta_120s",
    mechanism=(
        "Sustained touch-queue accumulation: the 120s delta of top-of-book "
        "quantity imbalance integrates REPEATED refill attempts rather than a "
        "single rebuild event. When the average touch imbalance drifts "
        "one-sided over two minutes, the queue is being strategically worked "
        "- iceberg-style replenishment keeping the displayed bid (or ask) "
        "heavy through successive consumptions, the passive leg of a "
        "meta-order establishing position. This slower touch trend is "
        "distinct from the 30s variant (single events) and from the deep "
        "depth5 deltas (levels 2-5), giving a third point in the top-vs-deep "
        "x fast-vs-slow decomposition of the live depth-momentum dimension."
    ),
    info_set="bid1_qty, ask1_qty",
    inspiration=(
        "iter-003 family brief: top-of-book vs deep-book decomposition; "
        "iceberg/hidden replenishment at the touch (Buti & Rindi 2013 "
        "undisplayed liquidity); iter-003 re-screen made depth MOMENTUM live "
        "(depth5_delta_60s PASS at 900s), the top-level analogue is untested."
    ),
    compute=compute,
)
