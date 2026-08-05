"""Explore-lane prototype spec (iter-003, depth-book lens).

wdi_accel_90s: second derivative of book state -- the delta of the 90s
(30-row) wdi delta. Acceleration of one-sided accumulation.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 30  # 30 x 3s rows = 90s momentum window; second diff spans 180s


def compute(part: pl.DataFrame) -> pl.Series:
    wdi = pl.col("wdi")
    delta = wdi.diff(D)          # 90s book-state momentum
    accel = delta.diff(D)        # change of that momentum (acceleration)
    return part.select(accel.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_accel_90s",
    mechanism=(
        "Second derivative of book state: how the 90s momentum of weighted "
        "depth imbalance is itself changing - the ACCELERATION of one-sided "
        "accumulation. Momentum says a build-up is under way; acceleration "
        "says whether it is intensifying or exhausting. Accelerating bid-side "
        "accumulation flags a fresh information episode still unfolding "
        "(continuation expected), while deceleration of a previously strong "
        "build-up marks exhaustion of the passive-flow source and precedes "
        "stall or reversion. Being a second difference, it is near-orthogonal "
        "to both the wdi level and the wdi momentum by construction, adding "
        "the turning-point dimension neither carries."
    ),
    info_set="wdi (library factor)",
    inspiration=(
        "iter-003 family brief: second-derivative extension of the live "
        "depth-momentum signal (wdi_mom_90s PASS at 900s); momentum-trigger "
        "acceleration logic (Cartea-Jaimungal-Penalva 2015 ch.8); iter-001 "
        "lesson: flow-MACD on already-windowed columns failed, but a second "
        "difference of a raw 3s state column is the correctly differentiated "
        "form."
    ),
    compute=compute,
)
