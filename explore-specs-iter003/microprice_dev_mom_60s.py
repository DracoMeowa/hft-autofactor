"""Explore-lane prototype spec (iter-003, depth-book lens).

microprice_dev_mom_60s: 60s momentum (delta) of the engine
microprice_dev -- changing queue-position pressure.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s delta


def compute(part: pl.DataFrame) -> pl.Series:
    return part.select(
        pl.col("microprice_dev").diff(D).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="microprice_dev_mom_60s",
    mechanism=(
        "Queue-position pressure momentum: microprice_dev is the "
        "quantity-weighted touch price minus the mid - it moves exactly when "
        "relative queue sizes at the touch shift. Its 60s delta separates "
        "sustained DRIFT of queue pressure (one-sided interest being queued, "
        "not yet executed) from single-tick noise. Rising microprice "
        "deviation momentum = the bid queue gaining weight at the touch, "
        "i.e. buy-side pressure building ahead of trades, predicting upward "
        "next moves. It is a derivative of a quantity that already embeds "
        "top-of-book position, distinct from oir/wdi momentum (ratio-based "
        "and deeper-weighted)."
    ),
    info_set="microprice_dev (library factor)",
    inspiration=(
        "iter-003 family brief: microprice leg of the depth-book family; "
        "Stoikov (2018) the micro-price as a queue-position predictor of the "
        "next price move; taking the TIME DERIVATIVE follows the meta-lesson "
        "that deltas carry what levels do not."
    ),
    compute=compute,
)
