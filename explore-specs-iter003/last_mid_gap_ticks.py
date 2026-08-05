"""Explore-lane prototype spec (iter-003, price-vol family).

last_mid_gap_ticks: raw aggressor-side signal (last_px - mid_px) in ticks.
Positive = last trade lifted the ask (buyer aggression); negative = hit the
bid (seller aggression); 0 = mid cross.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: SSE ETF minimum price increment (e.g. 588000): 0.001 RMB per tick
TICK = 0.001


def compute(part: pl.DataFrame) -> pl.Series:
    """(last_px - mid_px)/tick, current row only; defined wherever both exist."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    return part.select(gap_ticks.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="last_mid_gap_ticks",
    mechanism=(
        "Instantaneous aggressor side: the mid is the quote midpoint, so "
        "where the last trade landed relative to it reveals who crossed. "
        "last > mid means the last print lifted the ask (a buyer was the "
        "aggressor, paying up); last < mid means it hit the bid (seller "
        "aggression); last == mid is a mid cross. The magnitude in ticks "
        "also scales with the half-spread, so wide-quote aggression reads "
        "larger. The aggressor side is the fastest directional "
        "microstructure signal on the panel -- it is determined by the very "
        "trade that moves price -- and a fresh aggressor imbalance predicts "
        "immediate (15-60s) continuation before the book re-equilibrates."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 price-vol family brief: 'last_px vs mid_px encodes the "
        "aggressor side (which quote trades hit) -- usable via the panel "
        "alone'. Seed idea 8 (raw gap). Trade-direction / aggressor "
        "classification (Lee & Ready 1991); tick-rule aggressor pressure."
    ),
    compute=compute,
)
