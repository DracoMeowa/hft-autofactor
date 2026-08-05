"""Explore-lane prototype spec (iter-003, depth-book lens).

depth5_delta_120s: 120s change (delta) of the 5-level book imbalance --
slow window variant of depth5_delta_60s (eval-v2 PASS at 900s).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 40  # 40 x 3s rows = 120s delta


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
    name="depth5_delta_120s",
    mechanism=(
        "Slow depth momentum: the trailing-120s CHANGE in 5-level depth "
        "imbalance, integrating a full accumulation step of the deep book. "
        "Deep-book accumulation/distribution unfolds over minutes - meta-order "
        "slicing into passive limit orders, ETF creation/redemption inventory "
        "positioning - and a 60s delta sees only the first leg of such a "
        "build-up. The 120s window should match the 300-900s drift horizon "
        "where the parent depth5_delta_60s PASSED, testing whether longer "
        "integration improves horizon alignment and OOS stability of depth "
        "momentum, while differencing still strips the dead level state."
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 re-screen (2026-08-05): depth5_delta_60s PASSED eval-v2 at "
        "900s - depth momentum live; window grid above the PASS window targets "
        "the minutes-scale accumulation regime. Cao-Hansch-Wang (2009) book "
        "depth dynamics; Bouchaud et al. (2004) propagator tail justifies "
        "multi-minute integration of passive-flow state."
    ),
    compute=compute,
)
