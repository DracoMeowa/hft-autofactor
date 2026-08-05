"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

range_pos_x_wdi: champion interaction -- centered day-range position times
5-level depth imbalance. Book pressure means different things at different
heights of the day's battle range.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-12


def compute(part: pl.DataFrame) -> pl.Series:
    """(day_range_pos - 0.5) * wdi; warm-up rows null."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    cpos = pos - 0.5
    return part.select((cpos * pl.col("wdi")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_pos_x_wdi",
    mechanism=(
        "Depth imbalance conditioned on the range zone. A bid-heavy 5-level "
        "book (wdi > 0) near the day HIGH reads as queue build-up into "
        "resistance -- either genuine breakout intent (support under the "
        "ceiling push) or a passive trap for breakout chasers; the SAME "
        "bid-heavy book near the day LOW reads as defensive accumulation "
        "at support. Round 1's strongest admitted family was book-imbalance "
        "momenta (wdi/oir/depth deltas, 15-30s OOS IC 0.12-0.17), and the "
        "range-position champion was the only all-horizon, near-orthogonal "
        "source; multiplying them tests whether the book-pressure signal is "
        "amplified or flipped at the anchoring zones -- incremental to both "
        "parents, since each parent alone averages over the other dimension."
    ),
    info_set="mid_px, wdi",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 6 (champion x book "
        "imbalance interactions); combines the two strongest round-1 "
        "signal sources (wdi-momentum cluster and mid_day_range_pos)."
    ),
    compute=compute,
)
