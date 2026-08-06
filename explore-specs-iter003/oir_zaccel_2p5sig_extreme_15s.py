"""Explore-lane prototype spec (iter-003 R6C, family R6C).

oir_zaccel_2p5sig_extreme_15s: threshold sweep on the z-ACCELERATION-extremity
product. The 15s z-acceleration (2nd difference) of z_300(oir) weighted by
extremity (d2z * |z|), scored ONLY when |z| > 2.5. The oir base produced
the project's strongest |t| (oir_zaccel_extreme at 30s, t+27.74); this
tighter-gate variant tests whether the acceleration signal concentrates
further in the 2.5sigma+ tail of the touch-imbalance regime.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity and acceleration
THRESH = 2.5  # 2.5-sigma extremeness gate


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| when |z| > 2.5, else 0.0; warm-up rows null.

    The tight-extreme gate (2.5sigma) restricts the touch-imbalance
    acceleration-extremity product to the highest-conviction rows.
    """
    z = _z(pl.col("oir"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select(
        pl.when(z.is_null() | d2z.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z.abs() > THRESH)
        .then(d2z * z.abs())
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="oir_zaccel_2p5sig_extreme_15s",
    mechanism=(
        "Tail-isolated touch-imbalance acceleration: the 15s z-acceleration "
        "of z_300(oir) weighted by extremity (d2z * |z|), but scored ONLY "
        "when the regime stretch exceeds 2.5 sigma (|z| > 2.5, top ~1.2% "
        "of the z distribution), zeroed otherwise. The top-of-book "
        "imbalance is the most actively managed quote slot, and its "
        "z-acceleration (the curvature of the regime's z-trajectory, the "
        "project's strongest |t| at +27.74 in the ungated "
        "oir_zaccel_extreme_15s) isolates INTENSIFYING one-sided posting "
        "at the touch. The hypothesis under test: the ungated form fires "
        "on every row, including neutral-regime acceleration (routine "
        "quote maintenance); does the signal CONCENTRATE when the touch "
        "is already stretched beyond 2.5sigma -- the regime where the "
        "heavy best-quote side is pulling hard away from equilibrium and "
        "market makers are reposting at increasing urgency? If the 2.5sigma "
        "gate sharpens the IC, the bulk of the ungated acceleration signal "
        "lives in the tail and the neutral-regime acceleration is noise."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6C family brief: threshold sweep on the z-acceleration-"
        "extremity product; 2.5sigma on the oir base (round-5 strongest "
        "breakthrough) tests concentration in the tail."
    ),
    compute=compute,
)
