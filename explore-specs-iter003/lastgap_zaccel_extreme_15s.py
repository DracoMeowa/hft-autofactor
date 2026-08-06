"""Explore-lane prototype spec (iter-003 R6A, family R6A).

lastgap_zaccel_extreme_15s: z-ACCELERATION-extremeness product on the
last-trade-to-mid gap (ticks). The 15s acceleration (2nd difference) of
the 300s z-regime of the aggressor-side gap, weighted by the regime's
level extremity |z|. Mirrors the round-5 winning oir_zaccel template on
the trade-through base.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback for velocity and acceleration

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
    """d2z * |z| where d2z = 15s acceleration of the gap z.

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(_lastgap(), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="lastgap_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted aggressor-side stretch: the 15s "
        "acceleration (2nd difference) of z_300(last-mid gap in ticks), "
        "weighted by |z|. The gap is the fastest directional microstructure "
        "signal: positive = buyer lifted ask, negative = seller hit bid. "
        "Its z-acceleration isolates INTENSIFYING one-sided trade-through "
        "pressure from steady-state aggression: when the aggressor regime "
        "is already stretched versus its 300s norm (high |z|: sustained "
        "one-sided trade-through) and its curvature is accelerating "
        "further, the informed aggression is ramping up -- the rate of "
        "new marketable flow is itself increasing. Impact of freshly "
        "committed aggression is front-loaded: decisive acceleration of "
        "an already-stretched aggressor regime continues in its direction "
        "at 15-60s before the book absorbs. Economically distinct from "
        "lastgap_zvel_extreme_15s (round-5, velocity-weighted, dz * |z|): "
        "acceleration is the second derivative -- a steady high-velocity "
        "aggressor regime scores ~0 here (constant dz -> d2z~0); only "
        "regimes whose trade-through speed is itself changing fire."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R6A family brief: z-acceleration-extremeness template "
        "applied to the last-mid gap base NOT yet covered in acceleration "
        "form; the round-5 velocity product on this base was the strongest "
        "gap signal."
    ),
    compute=compute,
)
