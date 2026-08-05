"""Explore-lane prototype spec (iter-003, depth-book lens).

wdi_mom_180s: 180s momentum (delta) of the engine weighted depth
imbalance wdi -- slow window variant bracketing the PASS wdi_mom_90s
from above.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 60  # 60 x 3s rows = 180s delta


def compute(part: pl.DataFrame) -> pl.Series:
    return part.select(pl.col("wdi").diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_mom_180s",
    mechanism=(
        "Slow book-state momentum: the 180s change of the engine's weighted "
        "depth imbalance, integrating over intra-minute noise to measure the "
        "multi-minute ACCUMULATION REGIME of one-sided book build-up "
        "(meta-order passive legs, ETF basket arbitrage inventory). If depth "
        "momentum works at the 900s horizon because accumulation episodes "
        "persist, a longer delta should align the measurement with a fuller "
        "episode and improve OOS stability there; wdi's ~900s half-life "
        "means the 180s difference still removes the dead level state. "
        "Bracketing the PASS 90s window from above maps the horizon-response "
        "curve of book-state momentum."
    ),
    info_set="wdi (library factor)",
    inspiration=(
        "iter-003 re-screen (2026-08-05): wdi_mom_90s PASSED eval-v2 at 900s; "
        "window grid above the PASS window for horizon matching. Digest "
        "iter-000: wdi slowest-decaying signal on 588000 (IC 0.134/t=4.4 at "
        "900s); Cont-Kukanov-Stoikov (2014) book-event decomposition of "
        "queue changes."
    ),
    compute=compute,
)
