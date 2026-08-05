"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

hidden_imb_x_ti: hidden-layer imbalance x trade imbalance -- alignment of
patient queued depth (beyond top-5) with the aggressive flow direction
(hidden-absorption hypothesis).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def _hidden_imb() -> pl.Expr:
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
    """hidden-layer imbalance x trade_imbalance_60s."""
    ti = pl.col("trade_imbalance_60s")
    return part.select((_hidden_imb() * ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_x_ti",
    mechanism=(
        "Hidden-absorption alignment: the hidden-layer imbalance measures "
        "where PATIENT, deliberately-posted depth sits beyond the executable "
        "top-5; trade imbalance measures who is aggressing right now. Their "
        "product is a directional agreement score. Positive product = the "
        "queued hidden layer and the aggressive flow lean the SAME way: "
        "hidden depth is positioned to absorb and validate the flow, so the "
        "move has a resting reservoir behind it and tends to persist at "
        "15-60s. Negative product = aggression is attacking OPPOSITE the "
        "hidden reservoir: patient depth is absorbing the blows, blunting "
        "impact and setting up exhaustion/reversal of the aggressive side. "
        "This is the interaction member of the hidden-imbalance pair -- "
        "economically distinct from the hidden_imb_z level (which ignores "
        "flow) and bounded x bounded, so it cannot be a re-skin of either "
        "parent."
    ),
    info_set=(
        "total_bid_vol, total_ask_vol, depth_bid5, depth_ask5, "
        "trade_imbalance_60s (batch-2)"
    ),
    inspiration=(
        "iter-003 R3-B brief direction 5 (hidden-depth imbalance x trade-"
        "imbalance alignment); hidden absorption hypothesis; interactions "
        "passed where bare levels died in rounds 1-2."
    ),
    compute=compute,
)
