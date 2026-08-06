"""Explore-lane prototype spec (iter-003 R6, family R6B).

fullbook_imb_zvel_x_zaccel_15s: velocity-vs-acceleration AGREEMENT product
on the full-book imbalance (broadest depth state). z(15s z-velocity) crossed
with z(15s z-acceleration) of the patient deep-book positioning regime.
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
    """z(dz) * z(d2z) where dz = 15s z-velocity, d2z = 15s z-acceleration
    of the full-book imbalance z-regime.

    Warm-up rows null.
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
    return tmp.select((pl.col("_zdzz") * pl.col("_zd2zz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zvel_x_zaccel_15s",
    mechanism=(
        "Deep-book repositioning co-intensification: the product of the "
        "trailing-300s z of the 15s z-velocity of the full-book imbalance "
        "with the z of its own 15s z-acceleration. The full-book imbalance "
        "reaches beyond level 5, capturing patient institutional queue "
        "placement (creation/redemption flows); it builds slowly, so when "
        "its z-velocity and z-acceleration co-fire the SAME direction "
        "(product large positive), the broad positioning is being added to "
        "at an ACCELERATING rate -- a committed multi-level build, not a "
        "steady drift -- and the deep tilt continues at 15-60s. When the "
        "two oppose (product negative), the broad build is decelerating "
        "into exhaustion. Distinct from library fullbook_imb_zvel_div_15s "
        "(level-minus-velocity signed divergence) and "
        "fullbook_imb_zvel_extreme_15s (velocity x |level|): those involve "
        "the LEVEL; here both terms are derivatives, each normalized "
        "against its own 300s history, isolating pure derivative-on-"
        "derivative co-movement of the broadest book state."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6-B family brief: velocity-acceleration agreement product "
        "on the full-book imbalance base; the slow-build nature of deep-book "
        "positioning makes its acceleration a higher-conviction commitment "
        "signal than the touch."
    ),
    compute=compute,
)
