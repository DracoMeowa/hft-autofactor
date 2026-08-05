"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

hidden_imb_mom_300s: 300s momentum of the hidden-layer imbalance --
the multi-minute trend of patient queue positioning beyond the top 5.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 100  # 100 x 3s rows = 300s momentum window


def _hidden_imb() -> pl.Expr:
    """(hidden_bid - hidden_ask) / (hidden_bid + hidden_ask), clipped >= 0."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    hb = pl.when(tb - db > 0.0).then(tb - db).otherwise(pl.lit(0.0))
    ha = pl.when(ta - da > 0.0).then(ta - da).otherwise(pl.lit(0.0))
    den = hb + ha
    return (
        pl.when(den > 0.0)
        .then((hb - ha) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """300s delta of hidden-layer imbalance; warm-up rows null."""
    return part.select(_hidden_imb().diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_mom_300s",
    mechanism=(
        "Patient accumulation trend: the 300s change of the hidden-layer "
        "imbalance (full-book minus top-5 on each side) integrates a full "
        "multi-minute meta-order slicing cycle. Institutions splitting a "
        "parent order keep refreshing outer-level quotes for minutes while "
        "the touch only shows the current slice, so a sustained five-"
        "minute drift in hidden imbalance is the slow footprint of "
        "directional accumulation or distribution working through the deep "
        "queue. Unlike the 60s sibling (which catches fresh re-positioning "
        "events), this trend form measures commitment already accumulated, "
        "and such slow state should carry into 300-900s returns -- the "
        "horizon band where accumulation-style factors passed in the "
        "eval-v2 re-screen."
    ),
    info_set="total_bid_vol, total_ask_vol, depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 R2-C family brief direction 2 (hidden depth beyond the "
        "5 levels; structural slow variables at 300s); meta-order slicing "
        "and long impact tails (Bouchaud et al. 2009); eval-v2 re-screen "
        "lesson: accumulation/regime state lives at 900s."
    ),
    compute=compute,
)
