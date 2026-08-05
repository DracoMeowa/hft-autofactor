"""Explore-lane prototype spec (iter-003 R2-C, trade-structure lens).

trade_size_mom_60s: 60s change of log average trade size -- how fast
the tape is switching between retail-dust and institutional-ticket
granularity.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s momentum window


def compute(part: pl.DataFrame) -> pl.Series:
    """60s delta of log avg_trade_size_60s; warm-up rows null."""
    size = pl.col("avg_trade_size_60s")
    x = (
        pl.when(size > 0.0)
        .then(size.log())
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(x.diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="trade_size_mom_60s",
    mechanism=(
        "Granularity switching speed: the 60s change of log average trade "
        "size measures how fast the tape flips between retail-dust and "
        "institutional-ticket regimes. An abrupt within-minute size "
        "expansion flags block execution arriving NOW -- blocks cannot "
        "hide for long, so the switch itself is the event marker. On this "
        "growth ETF the block flow is hypothesized buy-skewed (creation-"
        "side baskets and accumulation programs lift in big tickets), so "
        "size acceleration should co-move mildly with subsequent buy-side "
        "pressure (positive IC at 15-60s); size contraction flags "
        "retailization and fading institutional presence. The log-differ-"
        "ence makes the measure scale-free across the intraday size "
        "regime, and the fast window targets the event horizon rather than "
        "the slow regime state measured by the 300s z sibling."
    ),
    info_set="avg_trade_size_60s (wishlist batch 1)",
    inspiration=(
        "iter-003 R2-C family brief direction 5 (trade granularity: "
        "avg_trade_size diff); block-arrival signature of granularity "
        "switches; Bouchaud et al. (2009) large-trade impact; main "
        "battleground horizons 15-60s per the family brief."
    ),
    compute=compute,
)
