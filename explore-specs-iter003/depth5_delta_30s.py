"""Explore-lane prototype spec (iter-003, depth-book lens).

depth5_delta_30s: 30s change (delta) of the 5-level book imbalance --
fast window variant of depth5_delta_60s, which PASSED the eval-v2
re-screen at 900s (2026-08-05).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 10  # 10 x 3s rows = 30s delta


def compute(part: pl.DataFrame) -> pl.Series:
    b = pl.col("depth_bid5").cast(pl.Float64)
    a = pl.col("depth_ask5").cast(pl.Float64)
    tot = b + a
    imb = (
        pl.when(tot > 0.0)
        .then((b - a) / tot)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(imb.diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="depth5_delta_30s",
    mechanism=(
        "Fast depth momentum: the trailing-30s CHANGE in 5-level depth "
        "imbalance (bid-ask)/(bid+ask), half the window of the eval-v2 PASS "
        "factor depth5_delta_60s. A 30s delta isolates the rapid reallocation "
        "of bid vs ask depth stock that precedes execution by seconds - "
        "informed traders pulling passive depth on the side they are about to "
        "trade against and stacking it on the side they favor - before the "
        "60s window averages in older, already-priced flow. The parent showed "
        "strong short-horizon OOS t at 15-60s but missed IS-retention; a "
        "faster window may retain the short-horizon information that decays "
        "inside the 60s integration, at the cost of more event noise."
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 re-screen (2026-08-05): depth5_delta_60s PASSED eval-v2 at "
        "900s with strong short-horizon OOS t failing only retention - depth "
        "momentum is live territory; window-grid exploration around a PASS "
        "factor. Cao-Hansch-Wang (2009) information content of book depth "
        "dynamics; meta-lesson: deltas/momenta, not levels, carry signal."
    ),
    compute=compute,
)
