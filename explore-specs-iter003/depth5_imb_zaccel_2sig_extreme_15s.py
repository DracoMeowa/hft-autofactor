"""Explore-lane prototype spec (iter-003 R6A, family R6A).

depth5_imb_zaccel_2sig_extreme_15s: strict-extreme gated z-acceleration on
the flat-weighted top-5 depth imbalance. d2z * |z| when |z| > 2.0, else 0.
Tests whether the visible-stack acceleration signal CONCENTRATES in the
2-sigma tail of the regime.
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
    """d2z * |z| when |z| > 2.0, else 0.0; warm-up rows null.

    The strict-extreme gate (|z| > 2) restricts the acceleration-extremity
    product to the highest-conviction regime rows.
    """
    z = _z(_depth5_imb(), W)
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
    name="depth5_imb_zaccel_2sig_extreme_15s",
    mechanism=(
        "Tail-isolated visible-stack acceleration: the 15s z-acceleration "
        "of z_300(flat-weighted top-5 depth imbalance) weighted by "
        "extremity (d2z * |z|), but scored ONLY when the regime stretch "
        "exceeds 2 sigma (|z| > 2.0, top ~5% of the z distribution), "
        "zeroed otherwise. Beyond 2 sigma the outer-visible-stack queue "
        "is genuinely crowded -- large resting orders across levels 3-5 "
        "tilted hard one way. Acceleration of that extreme (decisive "
        "rebuilding or abandonment of a stacked outer regime at "
        "increasing speed) continues at 15-60s; acceleration near a "
        "neutral top-5 imbalance is routine quote-maintenance noise. The "
        "strict gate isolates the highest-conviction rows, producing an "
        "event-sparse series distinct from the always-active "
        "depth5_imb_zaccel_extreme_15s. The economic question is signal "
        "CONCENTRATION: does the visible-stack acceleration product live "
        "entirely in the tail?"
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 R6A family brief: strict-extreme threshold variant of "
        "the z-acceleration-extremeness product on the top-5 imbalance "
        "base; tests signal concentration in the 2-sigma tail."
    ),
    compute=compute,
)
