"""Explore-lane prototype spec (iter-003 R6A, family R6A).

ti60_zaccel_x_15s: z-level crossed with acceleration DIRECTION on the 60s
trade imbalance. z_300(trade_imbalance_60s) * sign(d2z). The aggressive-
flow regime level gated by whether its own 15s acceleration confirms
(building) or opposes (exhausting) the level.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z * sign(d2z) where d2z = 15s acceleration of the trade-imbalance z.

    Warm-up rows null (z warm-up propagates through two shifts; sign of
    null acceleration is null, making the product null).
    """
    z = _z(pl.col("trade_imbalance_60s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((z * d2z.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti60_zaccel_x_15s",
    mechanism=(
        "Aggressive-flow level gated by acceleration direction: "
        "z_300(trade_imbalance_60s) * sign(d2z). The 60s taker imbalance "
        "z-level measures persistent one-sided aggression; its "
        "acceleration (d2z, the curvature of the z-trajectory) measures "
        "whether the aggression is intensifying or fading. When the level "
        "is extreme and acceleration confirms it (same sign), the "
        "aggressive program is still ramping -> continuation of the flow "
        "direction at 15-60s. When acceleration opposes the level, the "
        "aggression peaked and the overextended imbalance reverts. The "
        "binary sign(d2z) discards magnitude entirely, isolating inflection "
        "TIMING that velocity alone misses: a taker regime can have high "
        "positive velocity but zero acceleration (steady execution tempo, "
        "no new urgency) -- scored 0 here. Economically distinct from "
        "ti60_zaccel_extreme_15s (product form, retains magnitude) and "
        "from ti60_z_cross_vel_15s (round-4, velocity sign-flip event): "
        "this is the build-vs-exhaust curvature gate on the level."
    ),
    info_set="trade_imbalance_60s",
    inspiration=(
        "iter-003 R6A family brief: z-crossed-with-acceleration-sign "
        "construction on the trade-imbalance base; the acceleration "
        "direction gate is a different economic question than the product "
        "form on the same base."
    ),
    compute=compute,
)
