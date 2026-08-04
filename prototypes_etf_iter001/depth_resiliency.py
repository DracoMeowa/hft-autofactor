"""ETF-structure candidate 5: depth resiliency (total-depth replenishment).

After trades consume liquidity, market makers with quoting obligations
replenish the book.  The CHANGE in total depth (bid+ask), normalized by its
recent level, is a snapshot-computable resiliency proxy: rising total depth
signals active replenishment (shocks are transient -> mean reversion), while
falling total depth signals liquidity withdrawal (shocks are informative ->
continuation).  This is the depth-Delta dimension the digest lists as
unexplored, and it is constructed from TOTAL depth (bid+ask) rather than the
bid-vs-ask imbalance, so it is orthogonal to the saturated wdi/oir/microprice
mega-family.
"""
import polars as pl

#: 60s (20 x 3s rows) depth-change window
DIFF_ROWS = 20
#: trailing 300s (100 x 3s rows) level-normalization window
NORM_WINDOW = 100


def _compute(part: pl.DataFrame) -> pl.Series:
    tot = pl.col("depth_bid5").cast(pl.Float64) + pl.col("depth_ask5").cast(
        pl.Float64
    )
    dd = tot.diff(DIFF_ROWS)
    mtot = tot.rolling_mean(window_size=NORM_WINDOW, min_samples=NORM_WINDOW)
    frac = dd / mtot
    valid = mtot.is_not_null() & (mtot > 0.0) & frac.is_not_null()
    return part.select(
        pl.when(valid)
        .then(frac)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = {
    "name": "depth_resiliency",
    "mechanism": (
        "Depth resiliency: market makers with quoting obligations replenish "
        "the book after trades consume it. The 60s change in TOTAL depth "
        "(bid+ask), normalized by its trailing 300s level, is a snapshot "
        "resiliency proxy. Rising total depth = active replenishment, so "
        "recent shocks are transient (mean reversion); falling total depth "
        "= liquidity withdrawal, so shocks are informative (continuation). "
        "Built from total depth, NOT bid-vs-ask imbalance, so it sits in a "
        "different dimension from the wdi/oir/microprice mega-family."
    ),
    "info_set": "depth_bid5, depth_ask5",
    "inspiration": (
        "digest: 'Depth-side unexplored: resiliency, depth delta, queue "
        "position, large-order depth share'; Foucault-Kadan-Kandel (2005) "
        "and Obizhaeva-Wang (2013) resiliency -> reversal-vs-continuation "
        "over 1-5 minutes (targets the open 300s horizon)."
    ),
    "compute": _compute,
}
