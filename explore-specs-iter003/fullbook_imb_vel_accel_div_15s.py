"""Explore-lane prototype spec (iter-003 R6, family R6B).

fullbook_imb_vel_accel_div_15s: velocity-over-acceleration DOMINANCE ratio
on the full-book imbalance. z(15s z-velocity) / (1 + |z(15s z-accel)|): fires
when the broad deep-book positioning velocity is extreme but its acceleration
has faded -- a patient build whose driving curvature has gone.
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


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (sum); null if no book volume."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(dz) / (1 + |z(d2z)|): velocity-dominance over faded acceleration.

    Bounded ratio (denominator >= 1.0); warm-up rows null.
    """
    z_e = _z(_fullbook_imb(), W)
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
    name="fullbook_imb_vel_accel_div_15s",
    mechanism=(
        "Broad-book build overextension vs curvature fuel: the z of the 15s "
        "z-velocity of the full-book imbalance divided by (1 + |z of its 15s "
        "z-acceleration|). Deep-book positioning beyond level 5 reflects "
        "patient institutional queue placement and builds slowly; when its "
        "rebuild velocity reads regime-extreme but the acceleration has "
        "faded (denominator ~1), the broad positioning thrust has lost the "
        "curvature that was driving it -- a committed build that has spent "
        "its fuel and tends to stall/revert rather than continue. When "
        "acceleration is still strong the denominator grows, demoting "
        "still-fueled builds. Sign follows velocity. Distinct from "
        "fullbook_imb_zvel_x_zaccel_15s (multiplies the z's: acceleration "
        "amplifies) and from library fullbook_imb_zvel_div_15s (level minus "
        "velocity, both involving the level not two derivatives): the "
        "denominator construction isolates velocity-over-faded-curvature, "
        "the exhaustion regime -- the dual of co-intensification."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6-B family brief: velocity/acceleration ratio on the "
        "full-book imbalance; the slow-build nature of deep positioning "
        "makes curvature-fade a high-conviction exhaustion marker."
    ),
    compute=compute,
)
