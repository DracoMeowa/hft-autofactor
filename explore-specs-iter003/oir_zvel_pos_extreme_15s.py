"""Explore-lane prototype spec (iter-003 R6C, family R6C).

oir_zvel_pos_extreme_15s: SIGNED-extreme gated z-velocity on oir. The 15s
z-velocity of z_300(oir) weighted by the SIGNED stretch (dz * z), scored
ONLY when z > +2.0 (bid-heavy long-stretch). Isolates the long side of the
touch: tests whether velocity during bid-crowded regimes predicts
differently from the symmetric |z| form.
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

    The signed-long gate fires ONLY when the touch regime is stretched
    bid-heavy; velocity is amplified by the positive stretch magnitude z.
    """
    z = _z(pl.col("oir"), W)
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
    name="oir_zvel_pos_extreme_15s",
    mechanism=(
        "Long-stretch-isolated touch-imbalance velocity: the 15s z-velocity "
        "of z_300(oir) weighted by the SIGNED stretch (dz * z), but scored "
        "ONLY when the regime is stretched bid-heavy (z > +2.0), zeroed "
        "otherwise. When the best-quote imbalance (oir) reads beyond +2 "
        "sigma the heavy side of the touch is the BID -- aggressive "
        "informed buyers or passive institutional bid placement crowding "
        "the best bid. Velocity of that long-stretched touch (the rate at "
        "which the bid-heavy regime is being rebuilt or abandoned) "
        "captures informed BUYING pressure at the most liquid quote slot, "
        "and continues at 15-60s. The economic question is DIRECTIONAL "
        "ASYMMETRY: the round-5 admitted wdi_zvel_2sig_extreme and "
        "microprice_dev_zvel_2sig_extreme use the SYMMETRIC |z| gate "
        "(both tails); this spec isolates the long tail only. If the "
        "long-stretch IC differs materially from the short-stretch IC "
        "(see oir_zvel_neg_extreme_15s), the touch-velocity signal is "
        "directionally asymmetric -- on SSE ETFs, bid-side quote "
        "management (creation-driven) and ask-side (redemption-driven) "
        "operate through different mechanics."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6C family brief: signed-extreme gating tests "
        "directional asymmetry; this spec isolates the long-stretch "
        "(bid-heavy) tail of the oir velocity signal."
    ),
    compute=compute,
)
