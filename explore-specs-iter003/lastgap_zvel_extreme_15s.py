"""Explore-lane prototype spec (iter-003 R5, family R5-A).

lastgap_zvel_extreme_15s: z-level vs instantaneous-velocity divergence on
the last-trade-to-mid gap (ticks), PRODUCT form -- the 15s z-velocity of
the aggressor-side regime weighted by the extremity |z| of the regime.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback

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
    """dz * |z| where dz = 15s z-velocity of the aggressor-side regime.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(_lastgap(), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="lastgap_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted aggressor-side velocity: the 15s change rate "
        "of z_300(last-mid gap in ticks), weighted by |z|. The gap is "
        "the fastest directional microstructure signal: positive = buyer "
        "lifted ask, negative = seller hit bid. When the aggressor bias "
        "is already extreme versus its 300s norm (high |z|: sustained "
        "one-sided aggression -- a program executing persistently into "
        "one side) AND still accelerating in the same direction over "
        "15s, the informed flow is intensifying, not just persisting. "
        "Impact of freshly committed marketable flow is front-loaded: "
        "decisive acceleration of an already-stretched aggressor regime "
        "continues in its direction at 15-60s before the book absorbs "
        "and re-equilibrates. The extremity weight zeroes out routine "
        "aggressor noise around the norm. DEDUP: library "
        "last_mid_gap_ticks is the RAW instantaneous gap (single-row, "
        "no rolling context, no velocity); here the gap is z-scored over "
        "300s and the z-velocity is weighted by the level extremity. "
        "The library last_mid_gap_ma_30s is a moving-average smooth of "
        "the raw gap (a level-smoothing, not a z-vs-velocity "
        "construction). This is the only spec combining the gap's level "
        "extremity with its own z-velocity in regime-normalized form."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R5-A family brief: extreme (product) form of the "
        "ofi_z_cross_vel_15s template on the last-mid gap (constructed "
        "inline); round-4 showed the extreme construction was the "
        "goldmine."
    ),
    compute=compute,
)
