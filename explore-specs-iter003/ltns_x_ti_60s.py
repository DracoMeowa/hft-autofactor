"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ltns_x_ti_60s: signed large-trade direction CONFIRMED by aggregate
aggressive direction -- large_trade_net_share_60s x trade_imbalance_60s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """large_trade_net_share_60s x trade_imbalance_60s; null warm-up."""
    return part.select(
        (pl.col("large_trade_net_share_60s") * pl.col("trade_imbalance_60s"))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ltns_x_ti_60s",
    mechanism=(
        "Whale/crowd direction confirmation: the product of the SIGNED "
        "large-trade net share and the aggregate trade imbalance is large "
        "and positive only when the biggest tickets AND the overall "
        "aggressive flow point the same way. Same-sign confirmation marks "
        "institutional flow that is not hidden -- large informed traders "
        "sweeping in the direction the whole tape is already leaning -- "
        "the highest-conviction state, which continues at 15-60s. Opposite "
        "signs (whales one way, net aggression the other) mark absorption/"
        "iceberg behavior -- big players filling passively against the "
        "crowd -- a stall/reversal signature scored negative. Both terms "
        "are signed and bounded, so the interaction itself carries "
        "direction. Functionally distinct from the rejected round-1 "
        "ti_z_x_large_share (z x UNSIGNED share): here the large-trade "
        "term is signed, making the agreement/disagreement dimension "
        "visible rather than a level weight."
    ),
    info_set="large_trade_net_share_60s, trade_imbalance_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R2-B brief direction 4 (large-order direction x "
        "aggressive trade direction confirmation); large_trade_net_share_60s "
        "materialized 2026-08-06; Zhang & Shen (2018) trade-size "
        "decomposition of price impact."
    ),
    compute=compute,
)
