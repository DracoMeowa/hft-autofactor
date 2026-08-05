"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_dev_z_x_ti_60s: interaction of the deviation-regime z with the
current aggressive trade imbalance -- are EXECUTED aggressors confirming
or fighting the overextension?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s regime window on the deviation z


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(dev_from_open_bps, 300s) * trade_imbalance_60s; warm-up null."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    return part.select(
        (_z(dev, W) * pl.col("trade_imbalance_60s")).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="open_dev_z_x_ti_60s",
    mechanism=(
        "Executed-aggression confirmation of the anchor stretch. Unlike "
        "its order-book sibling (which tests QUEUE intent via OFI), this "
        "asks whether the trades actually hitting the tape in the last "
        "minute are on the side of the unusual deviation: positive "
        "product = aggressive buyers lifting while price is unusually "
        "stretched above the open (or sellers pressing while stretched "
        "below) -- real participation consuming liquidity in the "
        "extension direction, the footprint of informed continuation. "
        "Negative product = aggressors trading AGAINST the stretch: "
        "marketable flow is already harvesting the overextension, the "
        "mechanism through which anchored reversion physically happens "
        "(aggressive flow into the one-sided inventory restores the "
        "anchor). Trade imbalance is bounded [-1,1] and left un-z'd on "
        "purpose: it IS a signed pressure state, and multiplying by the "
        "deviation z gates it by 'is the stretch currently unusual', "
        "which is the condition under which aggressive flow is most "
        "diagnostic."
    ),
    info_set="mid_px, open_px, trade_imbalance_60s",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 5 (deviation x "
        "flow alignment) in the executed-flow lens; interaction-form "
        "lesson from rounds 1-2 (conditions > levels), with TI chosen "
        "over z(TI) because imbalance is already a bounded signed state."
    ),
    compute=compute,
)
