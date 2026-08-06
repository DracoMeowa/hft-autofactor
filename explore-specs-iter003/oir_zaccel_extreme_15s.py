"""Explore-lane prototype spec (iter-003 R5, family R5-B).

oir_zaccel_extreme_15s: NEW construction variant of the z-vs-velocity
template on oir -- ACCELERATION-extremity product. The 2nd difference
of z (15s acceleration of the z-regime) weighted by level extremity |z|.
Applied to top-of-book imbalance; tests acceleration-driven repositioning
at the touch.
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


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| where d2z = 15s acceleration of z.

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(pl.col("oir"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted touch-regime stretch: the 15s acceleration "
        "of z_300(oir) weighted by |z|. The top-of-book imbalance (best "
        "bid/ask qty ratio) is the most actively managed quote slot -- its "
        "z-acceleration (d2z, the curvature of the regime's z-trajectory) "
        "isolates INTENSIFYING one-sided posting from steady-state tilt. "
        "When the touch regime is already stretched (high |z|) and its "
        "rebuild is accelerating further, market makers are pulling and "
        "reposting the best quote at increasing speed -- an urgency signal "
        "that the queue tilt will continue at 15-60s. Distinct from "
        "oir_zvel_extreme_15s (round-4, velocity-weighted): acceleration is "
        "the second derivative, measuring whether the velocity itself is "
        "growing. A steady high velocity scores ~0 here (constant dz -> "
        "d2z~0); only changing-velocity regimes fire."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R5-B family brief: z-vs-acceleration construction on "
        "the oir base; tests whether the 2nd-derivative intensification "
        "pattern found in wdi also operates at the touch."
    ),
    compute=compute,
)
