"""Explore-lane prototype spec (iter-003 R6, family R6B).

wdi_vel_accel_div_15s: velocity-over-acceleration DOMINANCE ratio on the
5-level depth imbalance. z(15s z-velocity) / (1 + |z(15s z-accel)|): fires
when the multi-level queue rebuild velocity is extreme but its acceleration
has faded -- an overextended depth-imbalance thrust exhausting.
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
    """z(dz) / (1 + |z(d2z)|): velocity-dominance over faded acceleration.

    Bounded ratio (denominator >= 1.0); warm-up rows null.
    """
    z_e = _z(pl.col("wdi"), W)
    dz_e = z_e - z_e.shift(LAG)
    d2z_e = dz_e - dz_e.shift(LAG)
    tmp = part.select(
        z_e.alias("_z"), dz_e.alias("_dz"), d2z_e.alias("_d2z")
    )
    tmp = tmp.select(
        _z(pl.col("_dz"), W).alias("_zdzz"),
        _z(pl.col("_d2z"), W).alias("_zd2zz"),
    )
    return tmp.select(
        (pl.col("_zdzz") / (1.0 + pl.col("_zd2zz").abs())).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="wdi_vel_accel_div_15s",
    mechanism=(
        "Depth-rebuild overextension vs curvature fuel: the z of the 15s "
        "z-velocity of wdi divided by (1 + |z of its 15s z-acceleration|). "
        "Multi-level depth imbalance rebuilds more slowly than the touch "
        "and represents costlier, more committed positioning; when its "
        "rebuild velocity reads regime-extreme but the acceleration has "
        "faded (denominator ~1), the queue tilt was being driven hard but "
        "nothing is still pushing it -- an overextended depth thrust that "
        "has spent its momentum and is prone to revert. When the "
        "acceleration remains strong the denominator grows and shrinks the "
        "ratio, demoting the still-fueled rebuilds. The sign follows the "
        "velocity so the overextension direction is preserved. Distinct "
        "from wdi_zvel_x_zaccel_15s (multiplies the z's: acceleration "
        "amplifies) and from library wdi_zvel_extreme_15s / "
        "wdi_zaccel_extreme_15s (single derivative x level): acceleration "
        "in the DENOMINATOR answers the exhaustion question -- the dual of "
        "the co-intensification product."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6-B family brief: velocity/acceleration ratio on wdi; "
        "the multi-level queue's slower dynamics make acceleration-fade a "
        "cleaner exhaustion marker than at the touch."
    ),
    compute=compute,
)
