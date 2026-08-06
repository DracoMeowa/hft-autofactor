"""Explore-lane prototype spec (iter-003 R5, family R5-C).

wdi_zvel_x_spread_z: depth-imbalance extreme regime velocity gated by
spread-state stress -- the admitted wdi_zvel_extreme_15s base (15s z-velocity
of wdi weighted by regime extremity) multiplied by the 300s spread z.
Tests whether depth-imbalance velocity survives at 300-900s only under
spread stress (adverse-selection regime).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(dz_wdi * |z_wdi|) * z(spread, 300s); warm-up rows null."""
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    zvel = dz * z.abs()
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((zvel * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_x_spread_z",
    mechanism=(
        "Spread-stress-gated depth-imbalance velocity: wdi_zvel_extreme_15s "
        "is the strongest 15s IC signal in the library (extremity-weighted "
        "15s change rate of z_300(wdi)) but fades by 300-900s because much "
        "of that velocity is cheap routine queue churn in comfortable, "
        "tight-spread regimes that carries no long-horizon content. "
        "Hypothesis: when the SAME extreme-regime velocity occurs under "
        "WIDE, stressed spreads -- where market makers are fearful and "
        "adverse-selection risk is high -- rebuilding a crowded multi-level "
        "depth-imbalance queue is costly and likely informed, so the "
        "velocity direction continues at 300-900s. Multiplying by "
        "z(quoted_spread_ticks, 300s) gates magnitude AND flips sign across "
        "spread regimes: the product isolates the informed-stress component "
        "of the velocity, the part that survives at long horizons, and "
        "suppresses the routine-churn component. Different economic object "
        "from the bare zvel (averages over all spread regimes) and from "
        "div_z_x_spread_z (round-3: gated the hidden-depth divergence "
        "z-LEVEL, not a velocity of any book state)."
    ),
    info_set="wdi, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-C family brief: condition the round-4 z-velocity "
        "winner wdi_zvel_extreme_15s (15s IC +0.18, strongest short-"
        "horizon) on spread-z to extend its horizon; spread-z is the only "
        "consistently live interaction dimension across rounds 1-4."
    ),
    compute=compute,
)
