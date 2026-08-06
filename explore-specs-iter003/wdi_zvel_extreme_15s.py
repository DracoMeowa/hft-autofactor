"""Explore-lane prototype spec (iter-003 R4, family R4-C).

wdi_zvel_extreme_15s: z-level vs instantaneous-velocity divergence on wdi,
PRODUCT form -- the 15s z-velocity of the wdi regime weighted by the
extremity |z| of the regime being moved.
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
    """dz * |z| where dz = z_now - z_15s_ago; warm-up rows null."""
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted regime velocity: the 15s change rate of "
        "z_300(wdi), weighted by how extreme the regime being moved is "
        "(|z|). Fast motion OF an already extreme depth-imbalance regime "
        "is decisive -- a crowded multi-level queue state being rebuilt or "
        "abandoned in seconds -- and the direction of the motion continues "
        "at 15-60s; the same velocity around a neutral regime is routine "
        "queue churn and scores near zero because |z| is small. This is a "
        "level-x-velocity interaction with the direction carried by the "
        "velocity (odd under sign flip), not a level (round-1 lesson: slow "
        "levels dead) and not a raw momentum (library wdi_mom_* are "
        "unnormalized deltas): the extremity weight re-ranks velocity by "
        "the crowdedness of the state it moves, so it scores rows the raw "
        "momenta do not upweight."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R4-C family brief: product form of the admitted "
        "ofi_z_cross_vel_15s z-vs-velocity divergence template, made "
        "direction-carrying by weighting the z-velocity with the level "
        "extremity instead of gating to crossing events."
    ),
    compute=compute,
)
