"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

ofi_z_x_gap_fill_dir: order-flow regime z signed by the direction toward
the pre-close -- is queue pressure filling the overnight gap or fighting
the fill?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s regime window on OFI


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 300s) * sign(pre_close_px - mid_px); warm-up null."""
    toward_pc = (pl.col("pre_close_px") - pl.col("mid_px")).sign()
    return part.select(
        (_z(pl.col("ofi_60s"), W) * toward_pc).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_z_x_gap_fill_dir",
    mechanism=(
        "Is the order-book flow ENGINE of the gap-fill or its obstacle? "
        "The pre-close is the second anchor of the day; when mid sits "
        "above it, the fill direction is DOWN and positive value here "
        "means unusual sell-side queue pressure is actively pushing "
        "price toward the pre-close -- a genuine, flow-driven fill "
        "regime likely to persist until the anchor is reached (and "
        "possibly through it). Negative value with mid above the "
        "pre-close means queue pressure FIGHTS the fill: the gap-down "
        "pressure is being absorbed by bids, the fill is stalling, and "
        "the snap-back toward the open side is favored. The mirror holds "
        "below the pre-close. Gating OFI by the anchor-side sign turns a "
        "direction-free flow regime into a goal-directed pressure "
        "measure relative to the second anchor -- the question neither "
        "bare OFI z nor the gap-fill level can answer."
    ),
    info_set="ofi_60s, mid_px, pre_close_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 2 crossed with "
        "direction 5 (flow alignment vs the pre-close anchor); goal-"
        "directed flow conditioning in the spirit of anchored VWAP-"
        "reversion flow (Berkowitz, Logue & Noser 1988 on benchmark-"
        "driven trading)."
    ),
    compute=compute,
)
