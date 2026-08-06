"""Explore-lane prototype spec (iter-003 R5-D, spread raw-level wide gate).

div_vis_wide_walk: the admitted div_x_vis_share interaction (structural
mismatch x touch-concentration, both z-scored) active ONLY when the raw
spread exceeds its rolling mean -- fragile structural conflict gated to
the wide-quoting subset.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window and spread-mean window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _visible_share() -> pl.Expr:
    """(depth_bid5 + depth_ask5) / (total_bid_vol + total_ask_vol)."""
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
    """z(wdi - fbi, 300s) x z(visible_share, 300s) x 1{spread > mean}."""
    div_z = _z(pl.col("wdi") - _fullbook_imb(), W)
    conc_z = _z(_visible_share(), W)
    base = div_z * conc_z
    sp = pl.col("quoted_spread_ticks").cast(pl.Float64)
    sp_mean = sp.rolling_mean(window_size=W, min_samples=W)
    gate = (
        pl.when(sp_mean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(sp > sp_mean)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
    )
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="div_vis_wide_walk",
    mechanism=(
        "Concentration-unexplained mismatch active ONLY under wide "
        "quoting: the admitted div_x_vis_share base (z of touch-vs-"
        "queue divergence x z of touch-concentration) is zeroed unless "
        "the raw quoted spread exceeds its trailing-300s mean. The base "
        "captures the joint regime where a structural mismatch is large "
        "WHILE depth concentrates at the touch (hidden reserves too thin "
        "to explain the dislocation). Gating this to the wide-spread "
        "subset tests the acute-stress claim: the same fragile "
        "structural conflict resolves faster when quoting is "
        "expensive, because the cost to defend the dislocated posture "
        "is higher and arbitrageurs who would close the gap face a "
        "wider toll. Under comfortable sub-mean spreads the mismatch is "
        "buffered by cheap two-sided competition and is switched off. "
        "div_x_vis_share (round-3 admitted) has no spread interaction "
        "yet. The binary gate makes this an episode detector (exactly "
        "zero outside the wide regime), structurally different from the "
        "continuously signed base."
    ),
    info_set=(
        "wdi, total_bid_vol, total_ask_vol, depth_bid5, depth_ask5, "
        "quoted_spread_ticks"
    ),
    inspiration=(
        "iter-003 R5-D brief direction (a): wide_gate fill-in via a "
        "rolling-median threshold gate for the div_x_vis_share base "
        "(round-3 admitted, panel rho 0.074, no spread interaction yet)."
    ),
    compute=compute,
)
