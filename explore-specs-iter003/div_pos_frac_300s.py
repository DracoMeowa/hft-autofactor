"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

div_pos_frac_300s: trailing-300s fraction of snapshots where the top-5 vs
full-book divergence (wdi - full-book imbalance) is POSITIVE -- the
persistence/duration of the touch-leading-queue regime, an occupancy
statistic orthogonal to the magnitude z-score.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """Fraction of last-300s rows with (wdi - fbi) > 0; warm-up null."""
    div = pl.col("wdi") - _fullbook_imb()
    pos = (
        pl.when(div.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(div > 0.0)
        .then(1.0)
        .otherwise(0.0)
    )
    frac = pos.rolling_mean(window_size=W, min_samples=W)
    return part.select(frac.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="div_pos_frac_300s",
    mechanism=(
        "Regime DURATION, not magnitude: the share of the trailing 300s "
        "during which the executable 5-level imbalance (wdi) sits ABOVE the "
        "full-book imbalance. top5_book_div_z_300s measures how LARGE the "
        "touch-vs-queue gap is right now; this occupancy measures how LONG "
        "the touch has persistently led the deep queue. A persistently "
        "positive divergence is a durable structural posture -- displayed "
        "liquidity consistently out front of hidden liquidity -- which decays "
        "slowly and conditions the next minutes differently from a transient "
        "spike of identical magnitude. Duration is a different economic "
        "question than intensity: a small-but-persistent mismatch and a "
        "large-but-fleeting one hit the same z-extreme yet imply different "
        "commitment. Occupancy statistics are near-orthogonal to z-scores "
        "and to any imbalance momentum, sitting in the slow regime class "
        "that paid off at 300-900s in the eval-v2 re-screen."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R3-B brief direction 1 (persistence of divergence sign); "
        "top5_book_div_z_300s was round-2's strongest IC (15s OOS +0.214), "
        "this probes the duration dimension the z-score discards."
    ),
    compute=compute,
)
