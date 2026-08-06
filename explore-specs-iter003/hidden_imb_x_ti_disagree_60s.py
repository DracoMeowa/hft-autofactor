"""Explore-lane prototype spec (iter-003 R5-D, hidden-depth x trade flow).

hidden_imb_x_ti_disagree_60s: hidden-layer imbalance x trade imbalance,
ZEROED unless the two point OPPOSITE directions -- the ABSORPTION regime
where patient queued depth sits against the aggressive flow direction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def _hidden_imb() -> pl.Expr:
    """(hidden_bid - hidden_ask) / (hidden_bid + hidden_ask); null if no hidden depth."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    den = hb + ha
    return (
        pl.when(den > 0.0)
        .then((hb - ha) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """hidden_imb x trade_imbalance_60s, zeroed when signs agree; warm-up null."""
    product = _hidden_imb() * pl.col("trade_imbalance_60s")
    disagree = (
        pl.when(product.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(product < 0.0)
        .then(product)
        .otherwise(pl.lit(0.0))
    )
    return part.select(disagree.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_x_ti_disagree_60s",
    mechanism=(
        "Hidden-depth ABSORBS trade direction: the product of the "
        "hidden-layer imbalance and trade_imbalance_60s is kept ONLY "
        "when the two point opposite ways (negative product = "
        "disagreement), exactly zero otherwise. When patient queued "
        "depth sits OPPOSITE the aggressive flow, the hidden reservoir "
        "is absorbing the blows -- a resting liquidity wall blunting "
        "impact and setting up exhaustion/reversal of the aggressive "
        "side. A negative product (e.g. hidden-imbalance positive, ti "
        "negative) means hidden bids are absorbing selling: the "
        "sell-off is running into patient support and is prone to "
        "stall/revert. This is the DISAGREEMENT episode detector: it "
        "fires only on absorption rows and is silent during "
        "agreement. The raw product hidden_imb_x_ti (round-3 rejected) "
        "blended both regimes into one signed value and was IS-dead "
        "because they cancel in the IC; the agreement twin "
        "(hidden_imb_x_ti_agree_60s) isolates the continuation half. "
        "The two episode detectors are NOT sign-flips of each other: "
        "each fires on a disjoint subset of rows (agreement vs "
        "disagreement) and is exactly zero on the other subset, so "
        "their cross-sectional distributions have different support "
        "and carry genuinely different economic information."
    ),
    info_set=(
        "total_bid_vol, total_ask_vol, depth_bid5, depth_ask5, "
        "trade_imbalance_60s"
    ),
    inspiration=(
        "iter-003 R5-D brief direction (b): hidden imbalance x trade "
        "imbalance agreement/disagreement; the absorption hypothesis "
        "(patient depth against the flow = exhaustion/reversal) is the "
        "opposite prediction from the confirmation regime."
    ),
    compute=compute,
)
