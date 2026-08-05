"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

range_pos_delta_300s: 300s (100-row) DIFF of the position-in-day-range
(mid_day_range_pos, the round-1 champion).  How fast the battle is moving
through the intraday range -- the meta-lesson-compliant derivative of a
level factor.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100        # 100 x 3s rows = 300s delta window
EPS = 1e-12    # below this range the position is undefined (flat day)


def _range_pos() -> pl.Expr:
    """Position of mid within the causal intraday high-low range, [0,1]."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    return (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """range_pos(i) - range_pos(i-100); warm-up and flat-range rows null."""
    return part.select(_range_pos().diff(W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_pos_delta_300s",
    mechanism=(
        "The round-1 champion mid_day_range_pos passed ALL five horizons "
        "with negative IC (near the day high -> drift down: anchoring / "
        "stop-zone rejection dominates on this instrument), but the LEVEL "
        "is crowded as a library factor now; its 300s DELTA is the open "
        "derivative. A position rising fast over five minutes means the "
        "market is actively running at the day-high anchor zone, where "
        "the documented rejection regime bites hardest: fast approaches "
        "into the stop/resistance cluster predict the subsequent drift "
        "down (negative IC, like the level but on the CHANGE). A position "
        "falling fast = running at day-low support, predicting the mirror "
        "reaction. Differencing a bounded [0,1] state also strips the "
        "persistent anchoring component, so the delta decorrelates from "
        "the level by construction and from price momentum by "
        "normalization (it measures progress THROUGH the range, not "
        "returns: a small-range day and a large-range day with equal "
        "range-progress score equally)."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 5 (slow delta of the "
        "champion mid_day_range_pos, admitted round 1 with passes at all "
        "5 horizons, max library rho 0.41). Range-position recomputed "
        "inline from mid_px so the spec depends only on panel columns."
    ),
    compute=compute,
)
