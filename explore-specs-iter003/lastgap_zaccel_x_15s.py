"""Explore-lane prototype spec (iter-003 R6A, family R6A).

lastgap_zaccel_x_15s: z-level crossed with acceleration DIRECTION on the
last-trade-to-mid gap (ticks). z_300(last-mid gap) * sign(d2z). The
aggressor-side regime level gated by whether its own 15s acceleration
confirms (building) or opposes (peaking) the level.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback

#: SSE ETF minimum price increment (e.g. 588000): 0.001 RMB per tick
TICK = 0.001


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _lastgap() -> pl.Expr:
    """(last_px - mid_px) / TICK in ticks; positive = buyer aggression."""
    return (pl.col("last_px") - pl.col("mid_px")) / TICK


def compute(part: pl.DataFrame) -> pl.Series:
    """z * sign(d2z) where d2z = 15s acceleration of the gap z.

    Warm-up rows null (z warm-up propagates through two shifts; sign of
    null acceleration is null, making the product null).
    """
    z = _z(_lastgap(), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((z * d2z.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="lastgap_zaccel_x_15s",
    mechanism=(
        "Aggressor-side level gated by acceleration direction: "
        "z_300(last-mid gap in ticks) * sign(d2z). The gap is the fastest "
        "directional microstructure signal; its z-level measures persistent "
        "one-sided trade-through, and its acceleration (d2z, the curvature "
        "of the z-trajectory) measures whether the aggression is "
        "intensifying or fading. When the bullish gap regime is positive "
        "and acceleration is also positive, buyer aggression is still "
        "INTENSIFYING -- the rate of new marketable buys is curving up, "
        "and mid continues up at 15-60s. When the level is positive but "
        "acceleration turned negative, the aggression peaked and the "
        "overextended gap reverts as the book absorbs the last of the "
        "urgency. The binary sign(d2z) discards magnitude, isolating "
        "inflection TIMING that velocity (1st derivative) cannot answer: "
        "a trade-through regime can have high positive velocity but zero "
        "acceleration (steady aggression tempo, no new urgency) -- scored "
        "0 here. Economically distinct from lastgap_zaccel_extreme_15s "
        "(product form, retains magnitude) and from lastgap_zvel_extreme_15s "
        "(velocity-weighted): this is the curvature-direction gate on the "
        "level."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R6A family brief: z-crossed-with-acceleration-sign "
        "construction on the last-mid gap base; the acceleration direction "
        "gate is a different economic question than the product form."
    ),
    compute=compute,
)
