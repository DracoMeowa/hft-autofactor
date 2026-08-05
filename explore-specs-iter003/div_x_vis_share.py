"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

div_x_vis_share: top-5 vs full-book divergence z x book-concentration z --
a mismatch that stays large WHILE depth concentrates at the touch is an
acute, urgent dislocation not explained by hidden reserves.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _visible_share() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((db + da) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(wdi - fbi, 300s) x z(top-5 share of total depth, 300s)."""
    div_z = _z(pl.col("wdi") - _fullbook_imb(), W)
    conc_z = _z(_visible_share(), W)
    return part.select((div_z * conc_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="div_x_vis_share",
    mechanism=(
        "Concentration-unexplained mismatch: algebraically the divergence "
        "wdi - fbi equals hidden_share x (visible imbalance - hidden "
        "imbalance), so its magnitude is MECHANICALLY damped when depth "
        "concentrates at the touch (hidden_share low). A divergence that is "
        "nonetheless large WHILE the book is concentrated at the executable "
        "levels therefore implies an outsized visible-vs-hidden dislocation "
        "that hidden reserves cannot explain -- an acute, urgent structural "
        "conflict sitting right where trades execute, hypothesized to resolve "
        "directionally at 15-60s. Conversely, the same divergence in a deep, "
        "hidden-heavy book is buffered and benign. Multiplying the divergence "
        "z by the concentration z surfaces precisely the joint regime where "
        "the mismatch is large relative to the hidden depth available to "
        "produce it; product of two near-orthogonal z parents (one signed "
        "mismatch, one sign-blind shape)."
    ),
    info_set=(
        "wdi, total_bid_vol, total_ask_vol, depth_bid5, depth_ask5 (batch-2)"
    ),
    inspiration=(
        "iter-003 R3-B brief directions 1 x 3 (divergence x concentration); "
        "the decomposition divergence = hidden_share x (visible - hidden "
        "imbalance) motivates conditioning the mismatch on where depth sits."
    ),
    compute=compute,
)
