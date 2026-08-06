"""Explore-lane prototype spec (iter-003 R5-D, hidden-depth x trade flow).

hidden_imb_x_ti_agree_60s: hidden-layer imbalance x trade imbalance,
ZEROED unless both point the same direction -- the CONFIRMATION regime
where patient queued depth and aggressive flow agree.
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
    """hidden_imb x trade_imbalance_60s, zeroed when signs disagree; warm-up null."""
    product = _hidden_imb() * pl.col("trade_imbalance_60s")
    agree = (
        pl.when(product.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(product > 0.0)
        .then(product)
        .otherwise(pl.lit(0.0))
    )
    return part.select(agree.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_x_ti_agree_60s",
    mechanism=(
        "Hidden-depth CONFIRMS trade direction: the product of the "
        "hidden-layer imbalance and trade_imbalance_60s is kept ONLY "
        "when both point the same way (positive product = agreement), "
        "exactly zero otherwise. When patient queued depth behind the "
        "top 5 and aggressive trade flow lean the SAME direction, the "
        "move has a resting hidden reservoir behind it (hidden depth "
        "positioned to absorb and validate) -- the informed-continuation "
        "subset, expected to persist at 15-60s. This is the AGREEMENT "
        "episode detector: it fires only on confirmation rows and is "
        "silent (zero) during disagreement, structurally different from "
        "the raw product hidden_imb_x_ti (round-3 rejected) which "
        "blends agreement and disagreement into one signed value. The "
        "zero-mass disagreement half is the distinct economic input: "
        "round 3 found the raw product IS-dead because the two regimes "
        "(agreement = continue, disagreement = fade) have OPPOSITE "
        "signs and cancel in the IC; isolating the agreement half "
        "removes the cancellation. The disagreement half is the twin "
        "spec hidden_imb_x_ti_disagree_60s."
    ),
    info_set=(
        "total_bid_vol, total_ask_vol, depth_bid5, depth_ask5, "
        "trade_imbalance_60s"
    ),
    inspiration=(
        "iter-003 R5-D brief direction (b): hidden imbalance x trade "
        "imbalance agreement/disagreement; round-3 raw product died "
        "because the two sign-regimes cancel -- isolating one removes "
        "the cancellation."
    ),
    compute=compute,
)
