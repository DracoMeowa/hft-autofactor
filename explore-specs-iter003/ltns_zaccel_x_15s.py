"""Explore-lane prototype spec (iter-003 R6A, family R6A).

ltns_zaccel_x_15s: z-level crossed with acceleration DIRECTION on the
signed net share of large trades. z_300(large_trade_net_share_60s) *
sign(d2z). The institutional-footprint regime level gated by whether its
own 15s acceleration confirms (program building) or opposes (program
completing) the level.
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
    """z * sign(d2z) where d2z = 15s acceleration of the large-trade z.

    Warm-up rows null (z warm-up propagates through two shifts; sign of
    null acceleration is null, making the product null).
    """
    z = _z(pl.col("large_trade_net_share_60s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((z * d2z.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ltns_zaccel_x_15s",
    mechanism=(
        "Institutional-footprint level gated by acceleration direction: "
        "z_300(large_trade_net_share_60s) * sign(d2z). ltns carries the "
        "SIGNED direction of the largest ~10% of prints -- the whale "
        "footprint. Institutional execution programs build and complete "
        "over minutes, so the level's z measures whether a program is "
        "running, but its acceleration DIRECTION (binary sign of d2z, the "
        "curvature of the z-trajectory) isolates the program PHASE: when "
        "the bullish large-trade level is positive and acceleration is "
        "also positive, child-order tempo is still INTENSIFYING -- the "
        "buying program is ramping and mid continues up at 15-60s. When "
        "the level is positive but acceleration turned negative, the "
        "program is COMPLETING -- the last child orders are firing and "
        "directional pressure is about to cease, so price mean-reverts. "
        "The binary sign discards magnitude, isolating program-phase "
        "timing that velocity (1st derivative) misses: a whale regime "
        "can have high positive velocity but zero acceleration (steady "
        "execution tempo, neither ramping nor winding down) -- scored 0 "
        "here. Economically distinct from ltns_zaccel_extreme_15s "
        "(product form) and from ltns_zvel_div_60s (signed-divergence): "
        "this is the build-vs-complete curvature gate on the level."
    ),
    info_set="large_trade_net_share_60s (batch-2)",
    inspiration=(
        "iter-003 R6A family brief: z-crossed-with-acceleration-sign "
        "construction on the institutional-footprint base; the "
        "acceleration direction gate isolates execution-program phase, "
        "a different economic question than the product form."
    ),
    compute=compute,
)
