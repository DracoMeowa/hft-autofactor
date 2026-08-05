"""Explore-lane prototype spec (iter-003, price-vol family).

rv_ratio_60_300: volatility acceleration ratio rv_60s / rv_300s (relative
vol state, not unsigned level). Guarded against null/zero denominator.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: below this rv_300s magnitude the ratio is undefined (flat market)
EPS = 1e-14


def compute(part: pl.DataFrame) -> pl.Series:
    """rv_60s / rv_300s; null when rv_300s is null or ~0 (degenerate)."""
    num = pl.col("rv_60s")
    den = pl.col("rv_300s")
    ratio = (
        pl.when(den.is_not_null() & (den > EPS))
        .then(num / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(ratio.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="rv_ratio_60_300",
    mechanism=(
        "Volatility acceleration relative to its trailing state: the ratio "
        "of the trailing-60s realized variance to the trailing-300s "
        "realized variance measures whether variance is expanding NOW "
        "relative to the last five minutes. Ratio >> 1 flags a volatility "
        "burst (information arrival, stop cascade, quote withdrawal) and "
        "volatility clustering makes the burst persist, carrying both "
        "magnitude and -- via asymmetric/leverage effects -- directional "
        "drift into the next minutes. Ratio << 1 flags a volatility "
        "collapse (liquidity returning, calm before the next move). This "
        "is a RELATIVE vol form: the 300s denominator removes the slow "
        "baseline, so it is not the unsigned vol level that is library "
        "noise at short horizons."
    ),
    info_set="rv_60s, rv_300s (library)",
    inspiration=(
        "iter-003 price-vol family brief: 'rv_asym_300s PASSED -- the VOL "
        "family is live but only in signed/relative forms; the unsigned rv "
        "level is library noise'. Seed idea 5 (rv_60s/rv_300s). Vol "
        "clustering / persistence (Bollerslev 1986 GARCH; Andersen et al. "
        "2003 realized-vol dynamics)."
    ),
    compute=compute,
)
