"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

dev_from_open_bps: deviation of mid from the day's opening auction price,
in bps -- intraday stretch away from the first high-participation consensus.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """(mid - open_px) / open_px * 1e4; defined from the first row."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    return part.select(dev.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="dev_from_open_bps",
    mechanism=(
        "Stretch from the opening auction, in bps. The open is the day's "
        "first high-participation consensus price and a salient anchor for "
        "execution benchmarks and ETF creation/redemption arbitrage, which "
        "quote around open-referenced fair value. A mid stretched far from "
        "the open without fresh information accumulates reversion pressure: "
        "liquidity providers and arbitrageurs pull price back toward the "
        "anchored value, and inventory limits make one-sided drift away "
        "from the open increasingly expensive to sustain -- the further the "
        "deviation, the stronger the pull back. This is a whole-day "
        "ACCUMULATION state (not a rolling-window momentum): the anchor is "
        "fixed at 09:30 and the deviation integrates the entire session, "
        "which is exactly the slow-state horizon (300-900s) where round-1 "
        "winners lived."
    ),
    info_set="mid_px, open_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 5 (open/pre-close "
        "reference deviations) using the batch-2 open_px pass-through; "
        "opening-price anchoring (Tversky & Kahneman 1974) and intraday "
        "reversion to session anchors (Gao, Han, Li & Zhou 2018, market "
        "intraday seasonality)."
    ),
    compute=compute,
)
