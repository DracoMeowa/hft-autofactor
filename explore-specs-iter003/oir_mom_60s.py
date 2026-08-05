"""Explore-lane prototype spec (iter-003, depth-book lens).

oir_mom_60s: 60s momentum (delta) of the engine top-of-book order
imbalance ratio oir -- touch-level book-state derivative.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s delta


def compute(part: pl.DataFrame) -> pl.Series:
    return part.select(pl.col("oir").diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_mom_60s",
    mechanism=(
        "Touch-level book-state momentum: the 60s change of the engine's "
        "top-of-book order imbalance ratio (oir). The touch queue can turn "
        "independently of the deeper weighted book - fast queue events like "
        "sweeps and refill bursts hit level 1 first - so oir momentum is a "
        "distinct derivative from wdi momentum: it measures queue-position "
        "pressure changing at the EXECUTING level. Rising oir = the bid "
        "queue gaining relative weight at the touch, which in queue-reactive "
        "price impact pushes the next ticks up; differencing strips the oir "
        "level state that market makers have already priced."
    ),
    info_set="oir (library factor)",
    inspiration=(
        "iter-003 family brief: top-of-book vs deep-book decomposition of the "
        "live depth-momentum dimension; Cont-Stoikov-Talreja (2010) "
        "queue-reactive model - next-move probability is set by touch-queue "
        "evolution; wdi_mom_90s PASS at 900s motivates testing the touch "
        "analogue."
    ),
    compute=compute,
)
