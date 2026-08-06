"""Explore-lane prototype spec (iter-003 R6C, family R6C).

wdi_zvel_neg_extreme_15s: SIGNED-extreme gated z-velocity on wdi. The 15s
z-velocity of z_300(wdi) weighted by the SIGNED stretch (dz * z), scored
ONLY when z < -2.0 (ask-heavy short-stretch). Mirror of
wdi_zvel_pos_extreme_15s: isolates the short side of the multi-level depth
imbalance to test directional asymmetry.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback
GATE = -2.0  # short-stretch gate: z < -2.0


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * z when z < -2.0 (short-stretch), else 0.0; warm-up rows null.

    The signed-short gate fires ONLY when the depth-imbalance regime is
    stretched ask-heavy across levels 1-5.
    """
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    return part.select(
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z < GATE)
        .then(dz * z)
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_neg_extreme_15s",
    mechanism=(
        "Short-stretch-isolated depth-imbalance velocity: the 15s z-velocity "
        "of z_300(wdi) weighted by the SIGNED stretch (dz * z), but scored "
        "ONLY when the regime is stretched ask-heavy (z < -2.0), zeroed "
        "otherwise. The 5-level depth imbalance below -2sigma means the "
        "multi-level queue is CROWDED with ask liquidity -- large resting "
        "sell orders stacked across levels 1-5, a committed passive-selling "
        "state. Velocity of that short-stretched depth regime captures "
        "deep-book selling commitment that continues at 15-60s. The signed "
        "weighting amplifies by the negative stretch (z < -2), so the "
        "factor's sign convention differs from the symmetric |z| form. "
        "Paired with wdi_zvel_pos_extreme_15s (long-stretch), this spec "
        "tests DIRECTIONAL ASYMMETRY on the SAME wdi base: if the IC of "
        "the long-stretch subset and the short-stretch subset differ in "
        "magnitude or sign, the depth-imbalance velocity signal is "
        "asymmetric. On SSE ETFs, ask-side depth (redemption-driven "
        "passive selling) and bid-side (creation-driven) may carry "
        "different information content."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6C family brief: signed-extreme gating tests "
        "directional asymmetry; this spec isolates the short-stretch "
        "(ask-heavy) tail and pairs with wdi_zvel_pos_extreme on the "
        "same base for a clean asymmetry test."
    ),
    compute=compute,
)
