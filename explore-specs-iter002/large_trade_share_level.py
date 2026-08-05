"""Explore-lane prototype spec (iter-002, accumulation-regime lens).

large_trade_share_level: level of the engine's trailing-60s large-trade
volume share (wishlist column ``large_trade_share_60s``, materialized
2026-08-05).  First direct test of the size-distribution mechanism that
iter-001 could only approximate with snapshot deltas.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """Identity on the materialized column (engine trailing-60s share).

    The engine computes, per snapshot, the fraction of the trailing-60s
    traded volume carried by trades above the large-trade threshold.  Rows
    inside the 60s warm-up are null (engine convention), never zero-filled.
    """
    return part["large_trade_share_60s"]


PROTOTYPE = explore_prototype(
    name="large_trade_share_level",
    mechanism=(
        "Size-distribution level: a rising share of volume in large trades "
        "flags informed/institutional activity (stealth accumulation splits "
        "small orders, aggressive information events do not). Unlike ofi / "
        "trade_imbalance (signed net flow) this measures WHO is trading, not "
        "the net direction; on 15-60s horizons a burst of large prints "
        "typically precedes continuation of the just-revealed move, on "
        "300-900s it marks regime episodes. Direction-free by construction "
        "-- the sign comes from the concurrent price/flow columns if the IC "
        "is signed, otherwise the factor works via its own rank ordering."
    ),
    info_set="large_trade_share_60s (wishlist, materialized 2026-08-05)",
    inspiration=(
        "iter-001 candidates.json needs_materialization entry (accumulation-"
        "regime lens): trade-size distribution was untestable on the 12-"
        "column panel; digest: 'large prints concentrate information'. "
        "Zhang & Shen (2018, trade-size decomposition of price impact); "
        "Bouchaud et al. 2009 (anomalous impact of large trades)."
    ),
    compute=compute,
)
