"""Explore-lane prototype spec (iter-003, depth-book lens).

wdi_mom_30s: 30s momentum (delta) of the engine weighted depth
imbalance wdi -- fast window variant of wdi_mom_90s (eval-v2 PASS at
900s).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 10  # 10 x 3s rows = 30s delta


def compute(part: pl.DataFrame) -> pl.Series:
    return part.select(pl.col("wdi").diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_mom_30s",
    mechanism=(
        "Fast book-state momentum: the 30s change of the engine's weighted "
        "depth imbalance (wdi), faster than the PASS wdi_mom_90s. wdi "
        "weights depth by price distance from the touch, so its fastest "
        "movements reflect near-touch re-pricing of imbalance; a 30s delta "
        "captures abrupt one-sided build-ups before the 90s window smooths "
        "them away. The parent showed strong short-horizon OOS t at 15-60s "
        "while failing IS-retention; this variant tests whether book-state "
        "momentum has a distinct fast component that carries at 15-60s, "
        "where the short-horizon reward for fast-moving state lives."
    ),
    info_set="wdi (library factor)",
    inspiration=(
        "iter-003 re-screen (2026-08-05): wdi_mom_90s PASSED eval-v2 at 900s "
        "with strong short-horizon OOS t failing only retention; window grid "
        "below the PASS window. Digest iter-000: wdi is the cost-friendly "
        "champion with half-life ~900s, so even a 30s delta remains a genuine "
        "change-measurement far below the level's persistence."
    ),
    compute=compute,
)
