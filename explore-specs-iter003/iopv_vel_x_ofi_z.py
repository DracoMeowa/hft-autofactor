"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

iopv_vel_x_ofi_z: trailing-300s z-score of the product iopv_velocity x
ofi_60s -- joint episodes where the NAV drift and the ETF book flow agree
or disagree.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing reference window


def compute(part: pl.DataFrame) -> pl.Series:
    """Causal z of (iopv_velocity * ofi_60s) over 100 rows.

    Warm-up null; constant trailing windows (std == 0) map to 0.0.
    """
    x = pl.col("iopv_velocity") * pl.col("ofi_60s")
    mean = x.rolling_mean(window_size=W, min_samples=W)
    std = x.rolling_std(window_size=W, min_samples=W)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="iopv_vel_x_ofi_z",
    mechanism=(
        "The two legs of the ETF arbitrage chain are the NAV drift "
        "(iopv_velocity: where fair value is going) and the ETF's own "
        "book-building flow (ofi_60s: where limit-order pressure is "
        "pushing). When the two agree in sign and are jointly strong, the "
        "product is a large positive: arbitrageurs/informed traders are "
        "already positioning the ETF book in the direction of the moving "
        "anchor, so the mid continues that way at 60-300s. When they "
        "disagree (large negative product), ETF book flow is fighting the "
        "NAV drift; the anchor is the fundamental attractor, so the flow "
        "side tends to lose and the mid moves WITH the velocity, against "
        "the flow. Z-scoring the product against its trailing-300s "
        "distribution isolates unusual JOINT episodes from the everyday "
        "mix. This is a velocity-x-flow interaction on the batch-2 "
        "iopv_velocity column -- structurally different from the dead "
        "ofi_x_premium_sign / prem_x_ofi forms, which multiplied the "
        "regime-broken PREMIUM LEVEL by flow."
    ),
    info_set="iopv_velocity, ofi_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 1 (z of iopv_velocity x "
        "ofi_60s: change rate of arbitrage pressure). Round-1 lesson that "
        "state-conditioned interactions survive where levels die "
        "(ofi_z_x_spread_z passed while bare levels failed)."
    ),
    compute=compute,
)
