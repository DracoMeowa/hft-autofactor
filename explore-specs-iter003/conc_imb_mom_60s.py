"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

conc_imb_mom_60s: 60s momentum of the depth-CONCENTRATION asymmetry --
the bid vs ask share of each side's volume stacked in the executable
top 5 levels (liquidity at the head vs spread into the deep book).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s momentum window


def _conc_imb() -> pl.Expr:
    """depth_bid5/total_bid_vol - depth_ask5/total_ask_vol."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    cb = pl.when(tb > 0.0).then(db / tb).otherwise(pl.lit(None, dtype=pl.Float64))
    ca = pl.when(ta > 0.0).then(da / ta).otherwise(pl.lit(None, dtype=pl.Float64))
    return cb - ca


def compute(part: pl.DataFrame) -> pl.Series:
    """60s delta of concentration asymmetry; warm-up rows null."""
    return part.select(_conc_imb().diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="conc_imb_mom_60s",
    mechanism=(
        "Concentration asymmetry momentum: depth_bid5/total_bid_vol is "
        "the share of the bid side's volume stacked in the executable top "
        "5 -- visible urgency -- while a low share means the side is "
        "spread into patient deep levels. The bid-minus-ask difference of "
        "these shares measures WHICH side is moving liquidity toward the "
        "touch, and its 60s change captures active migration of placement "
        "strategy: bids migrating to the head while asks retreat deep "
        "means buyers urgent to execute against patient sellers, a "
        "configuration that precedes upward impact as the visible bid "
        "stack gets consumed and aggressively refilled. The construction "
        "is scale-free (shares, not volumes) and orthogonal to plain "
        "imbalance: a book can be balanced in total volume yet strongly "
        "asymmetric in where each side parks it."
    ),
    info_set="depth_bid5, depth_ask5, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R2-C family brief direction 3 (depth concentration: "
        "head vs deep placement and its two-side difference); book-shape "
        "distribution across levels (Zovio & Farmer 2002); placement-"
        "strategy migration as a flow-of-structure signal."
    ),
    compute=compute,
)
