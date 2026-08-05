"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ofi_ti_agree_60s: rolling sign-agreement share between book flow (OFI) and
aggressive trade imbalance -- passive and active flow pointing the same way
is a conviction signal.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 20  # 20 x 3s rows = 60s agreement window


def compute(part: pl.DataFrame) -> pl.Series:
    """share(sign(ofi)==sign(ti) != 0, 60s) - 0.5; warm-up rows null."""
    so = pl.col("ofi_60s").sign()
    st = pl.col("trade_imbalance_60s").sign()
    agree = pl.when(so.is_null() | st.is_null()) \
        .then(pl.lit(None, dtype=pl.Float64)) \
        .otherwise(
            pl.when((so == st) & (so != 0))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
        )
    share = agree.rolling_mean(window_size=W, min_samples=W)
    return part.select((share - 0.5).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_ti_agree_60s",
    mechanism=(
        "Cross-channel conviction: OFI (limit-side book building) and "
        "trade imbalance (aggressor-side execution) are partially "
        "independent information channels (rho ~ 0.5, not 1). When both "
        "point the same direction within the same minute, passive AND "
        "active participants agree -- high-conviction informed episodes "
        "that should continue at 15-60s. Persistent disagreement (book "
        "building one way while aggression runs the other) is absorption/"
        "iceberg behavior that precedes stalls or reversals. This is a "
        "SIGN-agreement statistic, deliberately complementary to "
        "flow_divergence_300s which compares z-scaled MAGNITUDES of the "
        "same two series."
    ),
    info_set="ofi_60s, trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 4; flow_divergence_300s champion "
        "proved the ofi/ti pair is the richest short-horizon info set; "
        "this probes agreement in the sign dimension instead of the "
        "magnitude dimension."
    ),
    compute=compute,
)
