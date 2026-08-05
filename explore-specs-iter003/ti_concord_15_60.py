"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ti_concord_15_60: cross-window CONSISTENCY of aggressive flow -- strength of
trade imbalance when the 15s and 60s windows point the SAME direction,
0 when they disagree. Bounded [-1,1], scale-free persistence filter.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """sign(TI_15s) x min(|TI_15s|,|TI_60s|) if same sign else 0.

    Uses min(|a|,|b|) = (|a|+|b|-|a-b|)/2, collapsing to 0 on opposite
    signs; single backward-looking combination, warm-up nulls propagate.
    """
    a = pl.col("trade_imbalance_15s")
    b = pl.col("trade_imbalance_60s")
    mag = (a.abs() + b.abs() - (a - b).abs()) / 2.0
    return part.select((a.sign() * mag).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_concord_15_60",
    mechanism=(
        "Cross-window aggression conviction: when the 15s and 60s trade-"
        "imbalance windows agree in sign, one-sided aggression is persistent "
        "from the quarter-minute to the full-minute scale -- a marketable "
        "program working through the book rather than ping-pong noise. The "
        "factor scores sign x min(|TI_15s|,|TI_60s|): capping at the weaker "
        "leg means conviction requires BOTH windows, and disagreement scores "
        "0. Both terms are bounded [-1,1], so the factor is scale-free "
        "across instruments and volume regimes. Persistent aggression "
        "depletes the touch queue and precedes 15-30s continuation in its "
        "sign. Active-channel mirror of ofi_concord_15_60; the two can "
        "diverge (aggression can persist while book building flips, marking "
        "liquidity being pulled while it is hit). A persistence filter, not "
        "an acceleration gap."
    ),
    info_set="trade_imbalance_15s, trade_imbalance_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R2-B brief direction 6 (cross-window same-sign strength); "
        "trade_imbalance_15s materialized 2026-08-06; persistence/conviction "
        "filter, scale-free active-channel counterpart of ofi_concord_15_60."
    ),
    compute=compute,
)
