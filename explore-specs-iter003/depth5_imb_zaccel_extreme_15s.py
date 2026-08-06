"""Explore-lane prototype spec (iter-003 R6A, family R6A).

depth5_imb_zaccel_extreme_15s: z-ACCELERATION-extremeness product on the
flat-weighted top-5 depth imbalance. The 15s acceleration (2nd difference)
of the 300s z-regime of the visible-stack imbalance, weighted by the
regime's level extremity |z|. Mirrors the round-5 oir_zaccel_extreme_15s
template on the unweighted top-5 base (levels 3-5 emphasized vs the
exp-weighted wdi).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback for velocity and acceleration


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _depth5_imb() -> pl.Expr:
    """(depth_bid5 - depth_ask5) / (sum); null when denominator is 0."""
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    den = db + da
    return (
        pl.when(den > 0.0)
        .then((db - da) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| where d2z = 15s acceleration of the top-5-imbalance z.

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(_depth5_imb(), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="depth5_imb_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted visible-stack regime stretch: the 15s "
        "acceleration of z_300(flat-weighted top-5 depth imbalance), "
        "weighted by |z|. The flat-weighted top-5 ratio (unlike the "
        "engine's exp-weighted wdi) is proportionally more sensitive to "
        "level 3-5 queue fills and pulls -- the outer visible stack where "
        "larger resting orders sit. Its z-acceleration (d2z, the curvature "
        "of the regime's z-trajectory) isolates INTENSIFYING one-sided "
        "visible-stack repositioning from steady-state tilt: when the "
        "outer-stack regime is already stretched (high |z|) and its "
        "rebuild is accelerating further, market makers are pulling and "
        "reposting the outer visible levels at increasing speed -- an "
        "urgency signal that the queue tilt continues at 15-60s. Distinct "
        "from depth5_imb_zvel_extreme_15s (round-4, velocity-weighted): "
        "acceleration is the second derivative. A steady high-velocity "
        "outer-stack regime scores ~0 here (constant dz -> d2z~0); only "
        "changing-velocity regimes fire."
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 R6A family brief: z-acceleration-extremeness template "
        "applied to the flat-weighted top-5 imbalance base; the round-4 "
        "velocity product on this base was admitted, so the 2nd "
        "derivative is the natural next economic question."
    ),
    compute=compute,
)
