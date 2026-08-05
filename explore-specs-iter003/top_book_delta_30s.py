"""Explore-lane prototype spec (iter-003, depth-book lens).

top_book_delta_30s: 30s delta of the TOP-of-book quantity imbalance
(bid1_qty - ask1_qty)/(bid1_qty + ask1_qty) -- queue rebuild at the
touch, the top-level decomposition complement to the 5-level depth
deltas.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 10  # 10 x 3s rows = 30s delta


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
    name="top_book_delta_30s",
    mechanism=(
        "Touch-queue rebuild dynamics: the 30s delta of top-of-book quantity "
        "imbalance. Level 1 is the only queue that actually executes, so its "
        "fast rebuild/depletion after trades is the fastest book event, "
        "decoupled from deeper-book positioning (which moves slower and is "
        "captured by the depth5 deltas). A bid1 queue actively rebuilding "
        "within 30s flags fresh passive demand arriving exactly at the price "
        "where the next tick forms; a depleting bid1 queue flags imminent "
        "downward queue exhaustion. Differencing the ratio removes the "
        "persistent queue-position state (the oir LEVEL, already priced), "
        "keeping only the flow into/out of the touch. Targets 15-30s "
        "queue-reactive horizons."
    ),
    info_set="bid1_qty, ask1_qty",
    inspiration=(
        "iter-003 family brief: top-of-book vs deep-book decomposition of the "
        "live depth-momentum signal; Cont-Stoikov-Talreja (2010) "
        "queue-reactive impact operates on the executing queue; queue events "
        "(sweeps, refills) are 3-15s phenomena per SSE L2 snapshot cadence."
    ),
    compute=compute,
)
