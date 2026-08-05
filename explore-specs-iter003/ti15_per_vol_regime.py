"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ti15_per_vol_regime: aggression PER UNIT OF VOLATILITY RISK -- raw
trade_imbalance_15s divided by the relative rv_60s regime (clipped to
[0.5, 2]). Trade-channel counterpart of ofi_per_vol_z_300s, but on the raw
fast level rather than a slow z.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s vol-regime window


def compute(part: pl.DataFrame) -> pl.Series:
    """trade_imbalance_15s / clip(rv_60s / mean(rv_60s, 300s), 0.5, 2).

    Warm-up rows null; the regime denominator is guarded (null unless the
    trailing vol regime is positive) so no inf/NaN path exists.
    """
    rv = pl.col("rv_60s")
    rv_regime = rv.rolling_mean(window_size=W, min_samples=W)
    rel = (
        pl.when(rv_regime.is_not_null() & (rv_regime > 0.0))
        .then(rv / rv_regime)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    rel_c = rel.clip(lower_bound=0.5, upper_bound=2.0)
    return part.select((pl.col("trade_imbalance_15s") / rel_c).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti15_per_vol_regime",
    mechanism=(
        "Aggression that arrives on a quiet tape is conspicuous: when "
        "realized volatility is BELOW its trailing-300s norm, one-sided "
        "marketable flow is rare and distinctive -- the footprint of an "
        "informed aggressor choosing a moment of low noise to work "
        "orders (Kyle 1985: informed flow hides in volume, not in "
        "volatility). The same imbalance during a vol burst is much more "
        "likely reactive -- stop runs, hedging feedback, panic "
        "liquidation -- and its continuation power is weaker. Dividing "
        "the raw 15s trade imbalance by the clipped relative-vol regime "
        "up-weights aggression in calm and discounts it in turbulence. "
        "Deliberately NOT another z-of-ti: ti_15s_z_120s died round 2 "
        "(rank-correlated with panel trade_imbalance_60s); the "
        "vol-regime denominator re-ranks rows across volatility states "
        "instead of across the flow's own history, which is the "
        "decorrelation lever. And not the dead regime_vol_x_flow "
        "either: that multiplied two 300s z-scores (rv_300s z x ti_60s "
        "z); this DIVIDES a raw fast imbalance by an rv_60s regime "
        "level ratio."
    ),
    info_set="trade_imbalance_15s, rv_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R3-C brief direction 3 (trade imbalance x volatility "
        "state); Kyle (1985) informed-flow timing; the round-2 death of "
        "ti_15s_z_120s shows the flow's own z is too close to panel "
        "trade_imbalance_60s -- conditioning on the vol regime is the "
        "open lane."
    ),
    compute=compute,
)
