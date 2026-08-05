"""Explore-lane prototype spec (iter-003, price-vol family).

signed_rv_60s: trailing-60s net signed realized variance -- rolling sum of
sign(ret) * ret^2 over 20 rows. Short-window, unnormalized cousin of the
PASSed rv_asym_300s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 20 rows x 3s = 60s trailing window
W = 20


def compute(part: pl.DataFrame) -> pl.Series:
    """sum_trailing( sign(ret) * ret^2 ); warm-up rows null."""
    ret = pl.col("mid_px").log().diff()
    signed_sq = ret.sign() * ret.pow(2)
    return part.select(
        signed_sq.rolling_sum(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="signed_rv_60s",
    mechanism=(
        "Directional concentration of variance over the last minute: the "
        "rolling sum of sign(ret)*ret^2 is up-RV minus down-RV in absolute "
        "(unnormalized) units. Positive means the quadratic variation of "
        "the last 60s is dominated by up moves (upside jumps), negative "
        "means down moves dominate (stop cascades / distressed selling). "
        "Unlike plain rv_60s it keeps the SIGN of which side the variance "
        "lives on, and unlike momentum it is variance-weighted, so a few "
        "large directional jumps dominate over many small ticks -- exactly "
        "the jump signature that carries. Being the 60s unnormalized cousin "
        "of the PASSed 900s rv_asym_300s, it targets the fast end of the "
        "same semivariance-asymmetry mechanism."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 price-vol family brief: 'rv_asym_300s PASSED at 900s -- "
        "vol family live in signed form'. Seed idea 7 (short-window cousin). "
        "Realized semivariance / good-bad volatility (Barndorff-Nielsen, "
        "Kinnebrock & Shephard 2010; Patton & Sheppard 2015)."
    ),
    compute=compute,
)
