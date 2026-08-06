"""Explore-lane prototype spec (iter-003 R6D, family R6D).

fullbook_imb_zaccel_15s_x_60s: cross-timescale ACCELERATION agreement on
the full-book imbalance (total_bid_vol vs total_ask_vol, all levels).
The 15s z-acceleration signed by the 60s z-acceleration direction,
weighted by regime extremity |z|.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG15 = 5
LAG60 = 20


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (sum); null when denominator is 0."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z15 * sign(d2z60) * |z| on the full-book imbalance z-regime.

    Warm-up rows null (z warm-up propagates through nested shifts).
    """
    z = _z(_fullbook_imb(), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    d2z15 = dz15 - dz15.shift(LAG15)
    d2z60 = dz60 - dz60.shift(LAG60)
    return part.select((d2z15 * d2z60.sign() * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zaccel_15s_x_60s",
    mechanism=(
        "Cross-timescale acceleration confirmation on the broad book: the "
        "15s z-acceleration of z_300(full-book imbalance) multiplied by "
        "the sign of the 60s z-acceleration, weighted by |z|. The "
        "full-book imbalance spans the ENTIRE visible book, not just five "
        "levels, so it captures passive institutional tilt in the outer "
        "queue that wdi cannot see. When its 15s and 60s curvature agree, "
        "the broad-book repositioning is intensifying at increasing speed "
        "across both horizons -- large passive positions being added or "
        "pulled at accelerating rate over both seconds and a minute, which "
        "is a stronger commitment signal than any single-scale acceleration. "
        "The |z| weight ensures only already-stretched broad regimes "
        "contribute. Economically distinct from fullbook_imb_vel15_x_vel60 "
        "(1st-derivative agreement) and fullbook_imb_zaccel_extreme_15s "
        "(single-scale 15s acceleration): this requires 60s corroboration "
        "of the acceleration direction, filtering out single-scale flicker."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6D family brief direction 1: cross-timescale "
        "acceleration agreement on the full-book imbalance base, which "
        "carries depth-beyond-5 information orthogonal to wdi."
    ),
    compute=compute,
)
