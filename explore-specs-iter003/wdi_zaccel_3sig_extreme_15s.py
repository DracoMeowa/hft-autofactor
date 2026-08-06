"""Explore-lane prototype spec (iter-003 R6C, family R6C).

wdi_zaccel_3sig_extreme_15s: threshold sweep on the z-ACCELERATION-extremity
product. The 15s z-acceleration (2nd difference) of z_300(wdi) weighted by
extremity (d2z * |z|), scored ONLY when |z| > 3.0. Tighter gate on the
acceleration template: beyond 3sigma the depth-imbalance regime is in rare
institutional-crowding territory, and acceleration there is the
highest-conviction commitment signal.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity and acceleration
THRESH = 3.0  # 3-sigma extremeness gate


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| when |z| > 3.0, else 0.0; warm-up rows null.

    The tight-extreme gate (3sigma) restricts the acceleration-extremity
    product to the rarest depth-imbalance regime rows.
    """
    z = _z(pl.col("wdi"), W)
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
    name="wdi_zaccel_3sig_extreme_15s",
    mechanism=(
        "Deep-tail-isolated depth-imbalance acceleration: the 15s "
        "z-acceleration of z_300(wdi) weighted by extremity (d2z * |z|), "
        "but scored ONLY when the regime stretch exceeds 3 sigma (|z| > "
        "3.0, top ~0.3% of the z distribution), zeroed otherwise. The "
        "z-acceleration (second difference of the z-regime) isolates "
        "INTENSIFYING one-sided depth posting -- rebuild speed that is "
        "itself accelerating, not steady. Beyond 3sigma the multi-level "
        "queue is in a genuine institutional-crowding state that only "
        "large committed positioning produces; acceleration there (the "
        "queue tilt's curvature growing sharper by the second) is the "
        "highest-conviction commitment signal -- market makers are "
        "reposting or pulling the stacked depth at increasing urgency. "
        "The tight gate tests whether the acceleration signal CONCENTRATES "
        "in the deepest tail, complementing the ungated "
        "wdi_zaccel_extreme_15s (which fires on every row). If the 3sigma "
        "gate retains or sharpens the signal, the bulk-regime acceleration "
        "is noise."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6C family brief: threshold sweep on the z-acceleration-"
        "extremity product (the round-5 breakthrough base); 3sigma tight "
        "gate on wdi tests concentration in the extreme tail."
    ),
    compute=compute,
)
