"""Explore-lane prototype spec (iter-003 R4, family R4-C).

oir_zvel_div_15s: z-level vs instantaneous-velocity divergence on oir,
SIGNED-DIFFERENCE form -- the slow touch-regime z minus its own fast
z-velocity, itself regime-normalized; overextension vs the fast edge.
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
    """z(oir, 300s) - z(dz, 300s) where dz = 15s z-velocity.

    Warm-up rows null: the z warm-up propagates through dz into the
    velocity's own trailing z. Both terms are regime-normalized, so the
    difference measures relative stretch, not raw momentum.
    """
    z_e = _z(pl.col("oir"), W)
    dz_e = z_e - z_e.shift(LAG)
    tmp = part.select(z_e.alias("_z"), dz_e.alias("_dz"))
    tmp = tmp.select(pl.col("_z"), _z(pl.col("_dz"), W).alias("_dzz"))
    return tmp.select((pl.col("_z") - pl.col("_dzz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zvel_div_15s",
    mechanism=(
        "Level-vs-velocity overextension at the touch: z_300(oir) minus "
        "the trailing-300s z of its own 15s z-velocity. When the slow "
        "touch regime reads extreme but the fast edge is moving the other "
        "way (large positive gap), the queue tilt is stretched against "
        "fresh quoting flow and drifts back toward the velocity-implied "
        "direction at 15-60s; when the normalized velocity leads the "
        "level (large negative gap), the regime is still building and "
        "continues. The fast edge leads the slow state: the gap closes "
        "through price. Both components are z-normalized against their "
        "own 300s distributions, so the object is a relative-stretch "
        "measure -- distinct from library oir_mom_60s (raw unnormalized "
        "delta) and from the dead bare oir level-z: velocity enters "
        "regime-relative and is SUBTRACTED, never multiplied or left raw."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R4-C family brief: signed-divergence form of the "
        "admitted ofi_z_cross_vel_15s z-vs-velocity template -- the "
        "tension between a slow regime level and its own fast "
        "instantaneous velocity, applied to top-of-book imbalance."
    ),
    compute=compute,
)
