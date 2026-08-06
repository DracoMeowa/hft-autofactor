"""Explore-lane prototype spec (iter-003 R6C, family R6C).

wdi_zvel_pos_extreme_15s: SIGNED-extreme gated z-velocity on wdi. The 15s
z-velocity of z_300(wdi) weighted by the SIGNED stretch (dz * z), scored
ONLY when z > +2.0 (bid-heavy long-stretch). Isolates the long side of
the multi-level depth imbalance.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback
GATE = 2.0  # long-stretch gate: z > +2.0


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * z when z > +2.0 (long-stretch), else 0.0; warm-up rows null.

    The signed-long gate fires ONLY when the depth-imbalance regime is
    stretched bid-heavy across levels 1-5.
    """
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    return part.select(
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z > GATE)
        .then(dz * z)
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_pos_extreme_15s",
    mechanism=(
        "Long-stretch-isolated depth-imbalance velocity: the 15s z-velocity "
        "of z_300(wdi) weighted by the SIGNED stretch (dz * z), but scored "
        "ONLY when the regime is stretched bid-heavy (z > +2.0), zeroed "
        "otherwise. The 5-level depth imbalance beyond +2sigma means the "
        "multi-level queue is CROWDED with bid liquidity -- large resting "
        "buy orders stacked across levels 1-5, a committed passive-buying "
        "state. Velocity of that long-stretched depth regime (the rate at "
        "which the bid-heavy stack is being rebuilt or withdrawn) captures "
        "deep-book buying commitment that continues at 15-60s. The economic "
        "question is whether the LONG-stretch subset of the admitted "
        "symmetric wdi_zvel_extreme_15s (dz * |z|, fires on both tails) "
        "carries the signal alone. If the short-stretch (ask-heavy) tail "
        "contributes little, the symmetric form is diluting its long-side "
        "signal with ask-side noise. Distinct from the oir pos variant: "
        "wdi aggregates 5 levels (broader queue state) vs oir's single "
        "best-quote slot, so the directional asymmetry may manifest "
        "differently across the depth."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6C family brief: signed-extreme gating tests "
        "directional asymmetry; this spec isolates the long-stretch "
        "(bid-heavy) tail of the wdi velocity signal across a different "
        "depth level than the oir variant."
    ),
    compute=compute,
)
