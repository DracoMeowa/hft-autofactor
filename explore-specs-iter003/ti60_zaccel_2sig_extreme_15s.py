"""Explore-lane prototype spec (iter-003 R6A, family R6A).

ti60_zaccel_2sig_extreme_15s: strict-extreme gated z-acceleration on the
60s trade imbalance. d2z * |z| when |z| > 2.0, else 0. Tests whether the
aggressive-flow acceleration signal CONCENTRATES in the 2-sigma tail.
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
    product to the highest-conviction aggressive-flow regime rows.
    """
    z = _z(pl.col("trade_imbalance_60s"), W)
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
    name="ti60_zaccel_2sig_extreme_15s",
    mechanism=(
        "Tail-isolated aggressive-flow acceleration: the 15s z-"
        "acceleration of z_300(trade_imbalance_60s) weighted by extremity "
        "(d2z * |z|), but scored ONLY when the regime stretch exceeds "
        "2 sigma (|z| > 2.0, top ~5% of the z distribution), zeroed "
        "otherwise. Beyond 2 sigma the 60s taker imbalance is a genuinely "
        "one-sided aggression regime -- sustained aggressive buying or "
        "selling well beyond the trailing norm, characteristic of a "
        "committed informed program. Acceleration of that extreme (the "
        "aggression curving further at increasing speed) continues to "
        "drive price at 15-60s; curvature near a neutral trade imbalance "
        "is routine two-sided flow noise. The strict gate isolates the "
        "~5% highest-conviction rows, producing an event-sparse series "
        "distinct from the always-active ti60_zaccel_extreme_15s. The "
        "economic question is signal CONCENTRATION: does the aggressive-"
        "flow acceleration product live entirely in the tail?"
    ),
    info_set="trade_imbalance_60s",
    inspiration=(
        "iter-003 R6A family brief: strict-extreme threshold variant of "
        "the z-acceleration-extremeness product on the trade-imbalance "
        "base; tests signal concentration in the 2-sigma tail."
    ),
    compute=compute,
)
