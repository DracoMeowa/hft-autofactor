"""Explore-lane prototype spec (iter-003, price-vol family).

rv_ratio_z_300s: causal trailing-300s z-score of the vol-acceleration ratio
rv_60s / rv_300s.  Distinguishes "vol is fast right now relative to its own
recent regime" from a persistently high baseline.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: below this rv_300s magnitude the ratio is undefined (flat market)
EPS = 1e-14
#: trailing 300s = 100 x 3s rows for the z-score window
W = 100


def _ratio() -> pl.Expr:
    """rv_60s / rv_300s, null on degenerate denominator."""
    num = pl.col("rv_60s")
    den = pl.col("rv_300s")
    return (
        pl.when(den.is_not_null() & (den > EPS))
        .then(num / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    r = _ratio()
    mean = r.rolling_mean(window_size=W, min_samples=W)
    std = r.rolling_std(window_size=W, min_samples=W)
    z = (r - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="rv_ratio_z_300s",
    mechanism=(
        "Vol-acceleration state, self-normalized: the trailing-300s "
        "z-score of the rv_60s/rv_300s ratio. The raw ratio (#rv_ratio) "
        "already removes the slow vol baseline; z-scoring it additionally "
        "removes the instrument/day-level typical RANGE of that ratio, so a "
        "value of +2 means variance is accelerating far beyond what this "
        "day's own recent regime has shown. That is a clean regime-shift "
        "trigger (sudden information arrival or liquidity evaporation) as "
        "opposed to 'vol has been elevated all day'. Regime shifts persist "
        "and carry directional drift, whereas the raw level of vol does not."
    ),
    info_set="rv_60s, rv_300s (library)",
    inspiration=(
        "iter-003 price-vol family brief: vol family live only in "
        "signed/relative forms. Seed idea 6 (causal z of rv_60s/rv_300s). "
        "Z-scored state convention of spread_z_300s; vol-of-vol regime "
        "shifts (Hamilton & Susmel 1999 ARCH-regime switching)."
    ),
    compute=compute,
)
