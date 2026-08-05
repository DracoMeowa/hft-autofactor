"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

vis_share_x_ti: book-concentration z x trade imbalance -- aggressive flow
striking a touch-concentrated (hidden-buffer-thin) book has amplified,
direction-persistent impact.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


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
    """z(top-5 share of total depth, 300s) x trade_imbalance_60s."""
    conc_z = _z(_visible_share(), W)
    ti = pl.col("trade_imbalance_60s")
    return part.select((conc_z * ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="vis_share_x_ti",
    mechanism=(
        "Concentration-gated flow impact: the same aggressive imbalance moves "
        "price very differently depending on where the book keeps its depth. "
        "When the top-5 share of total depth is unusually HIGH the hidden "
        "buffer behind the executable tip is thin, so directional trade flow "
        "that overwhelms the visible layer has nothing deep to absorb it and "
        "carries through -- continuation in the flow's direction. When depth "
        "is parked far from the touch (low concentration), hidden reserves "
        "replenish the consumed side and damp the move. Multiplying the "
        "concentration z by trade_imbalance_60s weights aggressive flow by "
        "the book's capacity to resist it: positive product = flow direction "
        "backed by a brittle, touch-heavy book. Sign-symmetric product of two "
        "near-orthogonal parents (one sign-blind shape, one signed flow), so "
        "it is not a re-skin of either."
    ),
    info_set=(
        "depth_bid5, depth_ask5, total_bid_vol, total_ask_vol, "
        "trade_imbalance_60s (batch-2 + library)"
    ),
    inspiration=(
        "iter-003 R3-B brief direction 3 x direction 4 (concentration regime "
        "gating flow); Kyle (1985) depth-inverse impact; state-conditioned "
        "interactions passed 15s in round 1 while bare levels died."
    ),
    compute=compute,
)
