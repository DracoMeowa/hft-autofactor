"""Explore-lane prototype spec (iter-003, depth-book lens).

book_slope_delta_60s: 60s change of the engine book_slope --
steepening/flattening dynamics of book shape.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s delta


def compute(part: pl.DataFrame) -> pl.Series:
    return part.select(pl.col("book_slope").diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="book_slope_delta_60s",
    mechanism=(
        "Book-shape dynamics: the 60s change of book_slope - whether the "
        "price-depth profile is actively STEEPENING or FLATTENING. The shape "
        "level says where the book is; its delta says where it is going. "
        "Active steepening toward one side means limit-order flow is "
        "currently building an asymmetric profile (fresh one-sided "
        "positioning -> continuation of that side), flattening means "
        "commitment is being withdrawn across levels (pressure dissipates). "
        "Shape can move without total bid/ask imbalance moving (depth "
        "migrating between levels), so this derivative is orthogonal to the "
        "wdi/depth5 deltas by construction."
    ),
    info_set="book_slope (library factor)",
    inspiration=(
        "iter-003 family brief: shape-dynamics leg of the depth-book family; "
        "meta-lesson: deltas/momenta are where signal lives - applied to the "
        "shape column no registered prototype has differentiated; Zovko & "
        "Farmer (2002) on book-profile dynamics."
    ),
    compute=compute,
)
