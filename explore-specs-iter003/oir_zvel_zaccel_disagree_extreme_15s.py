"""Explore-lane prototype spec (iter-003 R6, family R6B).

oir_zvel_zaccel_disagree_extreme_15s: opposite-sign-gated turning-point
detector on the top-of-book imbalance. z(velocity) x z(acceleration) is
scored ONLY when the two have OPPOSITE signs (the product is then negative,
large in magnitude only when both derivatives are co-extreme); exactly zero
on agreement. Sparse-ish event detector -- the disagreement twin of the
continuous agreement product.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity and acceleration


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(dz) * z(d2z), kept ONLY on opposite-sign rows (negative product);
    exactly 0.0 on agreement rows.

    The product magnitude is large only when both derivative z's are
    co-extreme, so the "both extreme" requirement is enforced continuously
    by the product rather than by a binary cutoff. Warm-up rows null.
    """
    z_e = _z(pl.col("oir"), W)
    dz_e = z_e - z_e.shift(LAG)
    d2z_e = dz_e - dz_e.shift(LAG)
    tmp = part.select(
        z_e.alias("_z"), dz_e.alias("_dz"), d2z_e.alias("_d2z")
    )
    tmp = tmp.select(
        _z(pl.col("_dz"), W).alias("_zdzz"),
        _z(pl.col("_d2z"), W).alias("_zd2zz"),
    )
    prod = pl.col("_zdzz") * pl.col("_zd2zz")
    return tmp.select(
        pl.when(pl.col("_zdzz").is_null() | pl.col("_zd2zz").is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(prod < 0.0)
        .then(prod)
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="oir_zvel_zaccel_disagree_extreme_15s",
    mechanism=(
        "Opposite-sign-isolated touch turning points: the product of the z "
        "of the 15s z-velocity of oir with the z of its 15s z-acceleration, "
        "kept ONLY when the two have OPPOSITE signs (product < 0), exactly "
        "zero on agreement. The product magnitude is large only when BOTH "
        "derivative z's are co-extreme, so the extremeness requirement is "
        "enforced continuously by the product itself, not a binary cutoff. "
        "A velocity pointing one way while an acceleration points the other "
        "is the textbook turning point: the touch tilt is still moving "
        "(extreme velocity) but its acceleration has flipped and is "
        "decelerating it (extreme opposing curvature) -- the rebuild is "
        "snapping direction against its own motion. The most-negative "
        "products are the highest-conviction deceleration episodes of an "
        "already-stretched touch regime; the velocity direction reverts at "
        "15-60s as the opposing curvature wins. Distinct from "
        "oir_zvel_x_zaccel_15s (the same product used EVERYWHERE, "
        "agreement-dominant): this twin fires only on the DISAGREE subset "
        "and is exactly zero elsewhere -- the two have disjoint support and "
        "carry genuinely different information (continuous co-movement vs "
        "turning-point episodes). Distinct from oir_zaccel_extreme_15s "
        "(acceleration x |level|): here acceleration is crossed with "
        "VELOCITY, not level, and gated to disagreement."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6-B family brief: vel-accel disagreement, gated to the "
        "extreme co-tail (opposite signs, both |z|>2), on the oir base; the "
        "disagreement episode detector is the turning-point dual of the "
        "agreement product (R5-D disagree twin structure proven on "
        "hidden_imb_x_ti_disagree_60s)."
    ),
    compute=compute,
)
