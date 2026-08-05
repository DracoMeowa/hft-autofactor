"""Explore-lane prototype spec (iter-003, price-vol family).

rv_z_300s: causal trailing-300s z-score of the library rv_300s.  Vol-state
relative to its own trailing distribution (a RELATIVE form, not the unsigned
level that is library noise).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 300s = 100 x 3s rows for the z-score window
W = 100


def compute(part: pl.DataFrame) -> pl.Series:
    x = pl.col("rv_300s")
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
    name="rv_z_300s",
    mechanism=(
        "Volatility regime-shift trigger, self-normalized: the trailing-"
        "300s z-score of the library rv_300s measures how far the current "
        "five-minute realized variance sits above/below ITS OWN recent "
        "distribution. The z removes the slow baseline and the day-level "
        "vol level, so a high value means volatility is spiking relative to "
        "what this day has shown so far -- a genuine regime shift "
        "(information arrival, liquidity evaporation), not merely an "
        "instrument that is volatile all day. Regime shifts persist via vol "
        "clustering and carry directional drift; the unsigned rv level does "
        "not, which is why this relative form is the live one."
    ),
    info_set="rv_300s (library)",
    inspiration=(
        "iter-003 price-vol family brief: 'vol family live in signed/"
        "relative forms; unsigned rv level is library noise'. Seed idea 15 "
        "(causal z of rv_300s). Z-scored state convention of spread_z_300s; "
        "vol regime switching (Hamilton & Susmel 1999)."
    ),
    compute=compute,
)
