"""Explore-lane prototype spec (iter-003, price-vol family).

vol_adj_mom_60s: last-minute momentum quality -- trailing-60s log-mid
momentum divided by sqrt(rv_60s).  A per-minute Sharpe of the move.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 20 rows x 3s = 60s trailing window
W = 20
#: below this rv the normalization is degenerate (no variance to scale by)
EPS = 1e-14


def compute(part: pl.DataFrame) -> pl.Series:
    """diff_20(log mid) / sqrt(rv_60s); null when rv is null or ~0."""
    mom = pl.col("mid_px").log().diff(W)
    vol = pl.col("rv_60s").sqrt()
    sharpe = (
        pl.when(pl.col("rv_60s").is_not_null() & (pl.col("rv_60s") > EPS))
        .then(mom / vol)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(sharpe.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="vol_adj_mom_60s",
    mechanism=(
        "Momentum quality rather than magnitude: dividing the trailing-60s "
        "log-mid return by the square root of the concurrent realized "
        "variance yields a per-minute Sharpe of the move. Two snapshots can "
        "show the same 60s return, but the one achieved with LOW variance "
        "(steady, one-directional drift) is the footprint of persistent "
        "informed flow and predicts continuation, while the one achieved "
        "with HIGH variance (choppy two-sided noise) is mean-reverting. "
        "Vol-adjustment separates signal drift from random-walk moves of "
        "equal size -- a distinct mechanism from both raw momentum and "
        "vol level."
    ),
    info_set="mid_px, rv_60s (library)",
    inspiration=(
        "iter-003 price-vol family brief seed idea 14 (20-row momentum / "
        "sqrt(rv_60s)). Momentum quality / signal-to-noise; analogous to "
        "Sharpe-normalized returns. Distinct from registered vol_adj_slope "
        "(which vol-adjusts book slope, not momentum)."
    ),
    compute=compute,
)
