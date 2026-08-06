"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

gap_vel_60s: 60s VELOCITY of the aggressor gap (last_px - mid_px in ticks)
-- trades walking the book vs the gap decaying.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment (588000)
D = 20        # 20 x 3s rows = 60s velocity step


def compute(part: pl.DataFrame) -> pl.Series:
    """diff( (last_px - mid_px)/tick, 20 ); first D rows null."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    return part.select(gap_ticks.diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="gap_vel_60s",
    mechanism=(
        "Aggressor-edge velocity: the gap between the last trade and the "
        "mid marks which side of the spread the tape is transacting on; "
        "its 60s CHANGE separates two regimes a level cannot. A RISING "
        "gap means successive prints are landing progressively further "
        "above the mid -- aggressors are walking THROUGH the quote, "
        "consuming the ask queue layer by layer (or the quotes have not "
        "yet repriced to the aggression): fresh, committed demand whose "
        "impact tends to continue at 15-60s. A SHRINKING gap means either "
        "the aggression is exhausting (prints drifting back toward mid) "
        "or quotes are catching up to a stale print -- the move is being "
        "digested. Because the gap decomposes as (last change) minus (mid "
        "change), its velocity nets out pure mid momentum and keeps the "
        "trade-vs-quote DIVERGENCE component: it is the aggressor walking "
        "the book relative to the quotes, not the price trending -- the "
        "economic question the admitted raw LEVEL (last_mid_gap_ticks, "
        "round-1 library factor) answers only at the current instant, not "
        "in motion."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R4-D brief direction (c) gap velocity (trades walking "
        "the book). Dedup note: last_mid_gap_ticks (raw level) is already "
        "a library factor -- velocity is its time derivative and a "
        "different economic question, but the pair must be watched by the "
        "dedup gate; also contains a -mid-momentum leg by construction."
    ),
    compute=compute,
)
