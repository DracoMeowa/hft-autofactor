"""Explore-lane prototype spec (iter-003 R6A, family R6A).

microprice_dev_zaccel_2sig_extreme_15s: strict-extreme gated z-acceleration
on microprice_dev. d2z * |z| when |z| > 2.0, else 0. Tests whether the
fair-value-lead acceleration signal CONCENTRATES in the 2-sigma tail.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| when |z| > 2.0, else 0.0; warm-up rows null.

    The strict-extreme gate (|z| > 2) restricts the acceleration-extremity
    product to the highest-conviction microprice-deviation regime rows.
    """
    z = _z(pl.col("microprice_dev"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select(
        pl.when(z.is_null() | d2z.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z.abs() > 2.0)
        .then(d2z * z.abs())
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="microprice_dev_zaccel_2sig_extreme_15s",
    mechanism=(
        "Tail-isolated fair-value-lead acceleration: the 15s z-"
        "acceleration of z_300(microprice_dev) weighted by extremity "
        "(d2z * |z|), but scored ONLY when the regime stretch exceeds "
        "2 sigma (|z| > 2.0, top ~5% of the z distribution), zeroed "
        "otherwise. Beyond 2 sigma the queue-weighted fair-value lead is "
        "genuinely stretched -- the heavy side of the book is pulling the "
        "microprice hard away from mid, an extreme positioning edge. "
        "Acceleration of that extreme (the queue imbalance curving further "
        "at increasing speed) continues to drag mid at 15-60s; curvature "
        "near a neutral micro-deviation is quote noise. The strict gate "
        "isolates the ~5% highest-conviction rows, producing an event-"
        "sparse series distinct from the always-active "
        "microprice_dev_zaccel_extreme_15s. The economic question is "
        "signal CONCENTRATION: does the fair-value-lead acceleration "
        "product live entirely in the tail?"
    ),
    info_set="microprice_dev",
    inspiration=(
        "iter-003 R6A family brief: strict-extreme threshold variant of "
        "the z-acceleration-extremeness product on microprice_dev; tests "
        "signal concentration in the 2-sigma tail."
    ),
    compute=compute,
)
