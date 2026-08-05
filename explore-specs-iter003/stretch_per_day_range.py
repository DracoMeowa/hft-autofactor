"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

stretch_per_day_range: deviation from the open normalized by the day's
running mid range -- stretch intensity per unit of demonstrated range.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-12  # below this range the ratio is undefined (flat day so far)


def compute(part: pl.DataFrame) -> pl.Series:
    """(mid - open)/(cum_max - cum_min); null while range ~0."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    out = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - pl.col("open_px")) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="stretch_per_day_range",
    mechanism=(
        "Stretch intensity priced in the day's own currency. A 20bp "
        "deviation from the open is routine on a day whose range is "
        "already 60bp (the whole session trended: deviation is earned, "
        "reversion pressure modest) but extreme on a day whose range is "
        "25bp (the deviation IS nearly the whole range: a one-sided "
        "stretch built on a thin, quiet tape, where anchored arbitrage "
        "and liquidity providers have the most to gain from snapping "
        "price back). Dividing by the running range normalizes the "
        "anchor pull by demonstrated volatility, bounding the value in "
        "[-1, 1] and making quiet-day and violent-day stretches "
        "comparable in one ranked column. Economically this is 'how "
        "much of the day's exploration budget has been spent on one "
        "side of the anchor' -- budget nearly exhausted on one side "
        "marks the overextension state where the round-2 open-deviation "
        "reversion IC is expected to bite hardest."
    ),
    info_set="mid_px, open_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 4 (deviation from "
        "open normalized by the day's running range); combines the two "
        "round-2 anchor constructs by DIVISION rather than blending "
        "them (ratio stays bounded and asks a distinct intensity "
        "question, mitigating the 0.767 parent-parent correlation)."
    ),
    compute=compute,
)
