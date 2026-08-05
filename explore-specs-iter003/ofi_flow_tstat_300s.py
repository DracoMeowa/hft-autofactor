"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ofi_flow_tstat_300s: signal-to-noise of the book channel -- trailing-300s
rolling mean of ofi_15s divided by its rolling std, the t-statistic of
book flow. Consistency per unit of own dispersion.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing t-stat window


def compute(part: pl.DataFrame) -> pl.Series:
    """rolling_mean(ofi_15s, 300s) / rolling_std(ofi_15s, 300s).

    Constant-dispersion windows map to 0.0 (neutral); warm-up rows null,
    never zero-filled.
    """
    x = pl.col("ofi_15s")
    m = x.rolling_mean(window_size=W, min_samples=W)
    s = x.rolling_std(window_size=W, min_samples=W)
    t = m / s
    return part.select(
        pl.when(s.is_not_null() & (s == 0.0))
        .then(pl.lit(0.0))
        .otherwise(t)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_flow_tstat_300s",
    mechanism=(
        "Flow consistency per unit of its own dispersion: two tapes can "
        "show the same AVERAGE book flow over five minutes -- one with "
        "steady one-sided building (low dispersion, high t-stat), the "
        "other with violent two-sided sloshing that averages out the "
        "same way (high dispersion, low t-stat). The first is the "
        "signature of committed, likely-informed queue investment whose "
        "drift continues at 60-900s (persistence = information; CKS "
        "2014); the second is churn whose average carries no commitment "
        "and mean-reverts. The rolling t-statistic of ofi_15s -- mean "
        "over std, both trailing 300s -- separates exactly these "
        "regimes and is signed by the flow direction. It is a different "
        "object from every registered OFI factor: levels and z-surprises "
        "measure instantaneous strength, accumulators sum magnitude, "
        "pos_frac counts signs -- none divides the flow's first moment "
        "by its second moment, so a noisy tape with strong average flow "
        "scores LOW here where magnitude factors score high."
    ),
    info_set="ofi_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R3-C brief direction 3 (flow-per-risk normalization, "
        "own-dispersion variant); CKS (2014) persistence-monotone OFI "
        "impact; round-2 lesson that slow horizons reward state "
        "statistics of flow quality rather than raw sums."
    ),
    compute=compute,
)
