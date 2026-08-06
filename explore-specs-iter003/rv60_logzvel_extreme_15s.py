"""Explore-lane prototype spec (iter-003 R5, family R5-A).

rv60_logzvel_extreme_15s: z-level vs instantaneous-velocity divergence on
log(rv_60s), PRODUCT form -- the 15s z-velocity of the vol regime weighted
by the extremity |z| of the regime being moved.
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


def _log_rv(x: pl.Expr) -> pl.Expr:
    """Safe log of rv: clips to positive minimum to avoid -inf on zero."""
    return x.cast(pl.Float64).clip(1e-20, 1e10).log()


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 15s z-velocity of the log(rv_60s) regime.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(_log_rv(pl.col("rv_60s")), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="rv60_logzvel_extreme_15s",
    mechanism=(
        "Extremity-weighted vol-regime velocity: the 15s change rate of "
        "z_300(log rv_60s), weighted by |z|. When 60s realized variance "
        "is already extreme versus its 300s norm (high |z|: either a vol "
        "spike or a vol lull) AND still accelerating in the same "
        "direction over 15s, the vol regime is being driven decisively "
        "-- either a full-blown information-event vol cascade building "
        "(positive z, positive dz) or a quiet-period unravelling into "
        "noise (negative z, positive dz). The SIGNED velocity "
        "distinguishes rising from falling vol: rising vol from an "
        "extreme (leverage effect: post-drop spike) predicts further "
        "short-horizon drift in the drop direction; falling vol from an "
        "extreme predicts stabilization. The extremity weight zeroes "
        "out routine vol noise around the norm. DEDUP: library rv_z_300s "
        "is the pure LEVEL z of rv_300s (slow window, no velocity); "
        "library signed_rv_60s is a from-scratch signed rv "
        "(sign(ret)*ret^2); here we z-score log(rv_60s) and weight the "
        "z-velocity by the level extremity. The log transform is the "
        "economic input change versus rv_z_300s (which uses raw rv_300s) "
        "-- it re-ranks the z by the proportional rather than absolute "
        "variance distance, preventing a single fat-tail spike from "
        "dominating the entire z distribution."
    ),
    info_set="rv_60s",
    inspiration=(
        "iter-003 R5-A family brief: extreme (product) form of the "
        "ofi_z_cross_vel_15s template on log(rv_60s); round-4 showed "
        "the extreme construction was the goldmine."
    ),
    compute=compute,
)
