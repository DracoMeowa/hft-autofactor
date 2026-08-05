"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

fullbook_imb_mom_60s: 60s momentum of the WHOLE-book bid/ask volume
imbalance (batch-2 total_bid_vol / total_ask_vol) -- a wider liquidity
pressure measure than the 5-level wdi.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s momentum window


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """60s delta of the full-book imbalance; warm-up rows null."""
    return part.select(_fullbook_imb().diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_mom_60s",
    mechanism=(
        "Full-book pressure momentum: batch-2 total_bid_vol/total_ask_vol "
        "sum the ENTIRE visible book, not just five levels, so their "
        "normalized imbalance measures queued liquidity pressure across "
        "all prices, with the outer queue included. The 60s CHANGE of this "
        "full-book imbalance isolates fresh one-sided queue build-up or "
        "withdrawal happening now: a rising value means resting interest "
        "is arriving faster on the bid side at every level, which precedes "
        "upward queue-reactive impact as aggressors consume the thinning "
        "ask stack (Cont-Stoikov-Talreja pressure logic, generalized from "
        "5 levels to the whole book). The momentum form strips the slowly "
        "varying level regime (the dead-end class per the iter-001/002 "
        "meta-lesson) and sits in the book-imbalance-momentum cluster that "
        "was round-1's strongest short-horizon family, but on a WIDER base "
        "than wdi/oir/depth5 deltas, so it is not a re-run of them."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R2-C family brief direction 1 (full-book imbalance and "
        "its diff); queue-reactive pressure (Cont, Stoikov & Talreja "
        "2010) generalized beyond 5 levels; batch-2 total_bid_vol/"
        "total_ask_vol materialized 2026-08-06; round-1 strongest cluster "
        "= fast book-imbalance momenta at 15-60s."
    ),
    compute=compute,
)
