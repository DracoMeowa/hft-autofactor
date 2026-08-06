"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

gap_pin_persist_300s: trailing-300s signed OCCUPANCY of the aggressor gap
-- how persistently prints pin one side of the mid (one-sided pinning).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment (588000)
W = 100       # 100 x 3s rows = 300s trailing window


def compute(part: pl.DataFrame) -> pl.Series:
    """rolling_mean( sign((last-mid)/tick), 300s ); warm-up rows null."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    pinned = gap_ticks.sign()
    return part.select(
        pinned.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="gap_pin_persist_300s",
    mechanism=(
        "One-sided pinning duration: when the last trade keeps landing on "
        "the SAME side of the mid for most of the trailing 300s, the tape "
        "is pinned -- demand (or supply) is repeatedly crossing the spread "
        "on one side and the quote has not been allowed to re-center. "
        "Persistent one-sided pinning is the footprint of a committed "
        "directional participant working an order (or of one-sided "
        "inventory pressure from the quoting side): a durable state that "
        "decays slowly and conditions drift over minutes, not seconds. "
        "Signed occupancy (mean of the gap sign) separates DURATION from "
        "INTENSITY: a small-but-persistent pin and a large-but-fleeting "
        "aggressor print are the same z-extreme of the raw level yet "
        "imply different commitment -- the duration-vs-intensity split "
        "that made div_pos_frac_300s and hidden_imb_pos_frac_300s live in "
        "round 3. Mid-cross prints (sign 0) dilute the occupancy toward "
        "neutral. Different question from the admitted raw LEVEL "
        "(last_mid_gap_ticks): this is how LONG the pinning has lasted."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R4-D brief direction (c) persistence of one-sided "
        "pinning; round-3 pos-frac/duration template (div_pos_frac_300s "
        "admitted) applied to the aggressor gap sign."
    ),
    compute=compute,
)
