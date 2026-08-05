"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

overnight_gap_x_range_pos: champion interaction -- overnight gap
(open vs pre-close) times centered day-range position. Overnight context
disambiguates what the same intraday position means.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-12


def compute(part: pl.DataFrame) -> pl.Series:
    """((open - pre_close)/pre_close) * (day_range_pos - 0.5)."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    gap = (pl.col("open_px") - pl.col("pre_close_px")) / pl.col("pre_close_px")
    return part.select((gap * (pos - 0.5)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="overnight_gap_x_range_pos",
    mechanism=(
        "Overnight context disambiguates intraday range position. The gap "
        "(open vs previous close) is the overnight sentiment repricing; the "
        "SAME position near the day's high means different things under "
        "different overnight backdrops: on a gap-UP day, sitting at the day "
        "high means overnight gains are being HELD and extended (two "
        "timeframes agreeing, continuation favored); on a gap-DOWN day, a "
        "day-high print means the entire overnight loss has already been "
        "recovered intraday (exhausted catch-up, rejection favored). The "
        "gap alone is constant within a day (no intraday rank information), "
        "and position alone ignores the overnight state; the product varies "
        "within the day and lets the anchoring signal flip sign with the "
        "overnight regime."
    ),
    info_set="mid_px, open_px, pre_close_px",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 5 (open/pre-close "
        "references) crossed with direction 6 (champion interactions), "
        "using the batch-2 open_px / pre_close_px pass-throughs; overnight-"
        "gap conditioning of intraday reversals (Lou, Polk & Skouras 2019 "
        "on overnight/inside-day return interplay)."
    ),
    compute=compute,
)
