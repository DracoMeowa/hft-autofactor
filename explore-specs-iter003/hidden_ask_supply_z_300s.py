"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

hidden_ask_supply_z_300s: trailing-300s z-score of the ASK-side hidden
share -- the fraction of total ask depth parked BEYOND the executable
top-5 levels. Hidden ask supply regime.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(hidden_ask / total_ask, 300s); warm-up rows null."""
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    share = (
        pl.when(ta > 0.0)
        .then(ha / ta)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(_z(share, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_ask_supply_z_300s",
    mechanism=(
        "Hidden ask supply regime: the fraction of the ENTIRE ask side that "
        "rests beyond the executable top-5 levels, z-scored against its own "
        "trailing-300s distribution. The supply-side twin of hidden bid "
        "support: an ask side whose size is parked deep is patient SELLING "
        "interest stacked above the touch -- distribution queued out of sight "
        "that keeps refilling the visible offer and caps rallies / forms a "
        "ceiling. An ask side thin behind the executable tip offers little "
        "latent supply, so consumption of the visible offer is more likely to "
        "walk price up. Decomposing the book by SIDE exposes where hidden "
        "liquidity sits that a net imbalance washes out; z-scoring keeps the "
        "measure in the live regime class rather than the dead raw-level "
        "class. Distinct from conc_imb_z (bid-vs-ask concentration asymmetry) "
        "because this is a one-sided absolute of hidden placement."
    ),
    info_set="depth_ask5, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 2 (side decomposition: hidden ask "
        "supply); displayed vs undisplayed reserves (Buti & Rindi 2013); "
        "depth beyond level-5 confirmed as real incremental info in round 2, "
        "but only in ratio/z form."
    ),
    compute=compute,
)
