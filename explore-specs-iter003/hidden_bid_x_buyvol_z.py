"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_bid_x_buyvol_z: hidden bid-support z x z(buy_vol_60s) -- patient bid
reservoirs ALIGNED with aggressive buying (demand-regime confirmation), both
parents regime-normalized.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window on both parents


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _hidden_bid_share() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    return (
        pl.when(tb > 0.0)
        .then(hb / tb)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(bid hidden share, 300s) x z(buy_vol_60s, 300s); warm-up null."""
    hb_z = _z(_hidden_bid_share(), W)
    buy_z = _z(pl.col("buy_vol_60s").cast(pl.Float64), W)
    return part.select((hb_z * buy_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_bid_x_buyvol_z",
    mechanism=(
        "Demand-regime confirmation across channels: the product of the "
        "bid-side hidden-support z and the z of SIDE-ATTRIBUTED buy volume. "
        "When both are high, aggressive buying is happening WHILE an "
        "unusually deep patient-bid reservoir sits below the touch -- the "
        "visible aggression has hidden demand stacked behind it, the "
        "iceberg-style replenishment that lets buying pressure persist "
        "instead of exhausting on the visible tip, so continuation at "
        "15-60s. When buy flow runs with a SHALLOW hidden bid reservoir, "
        "the aggression is exposed -- nothing queued to renew the bid once "
        "the touch is consumed -- and the move is fragile. The round-3 "
        "hidden-x-flow attempts all used NET trade imbalance (bid_absorb_x_"
        "sell = z(hidden bid) x -TI, hidden_imb_x_ti = hidden imb x TI) "
        "and died; the economic input here is different: one-sided GROSS "
        "buy volume, which stays high in two-sided heavy trading where net "
        "TI is ~0, so this interaction fires in exactly the regimes the "
        "dead TI-products cannot see. Both parents z-scored: product of two "
        "near-orthogonal regime coordinates, bounded away from raw-scale "
        "artifacts."
    ),
    info_set="total_bid_vol, depth_bid5, buy_vol_60s (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (c): hidden_bid_support x buy_vol_60s "
        "alignment in z-form; side-attributed batch-2 volumes as the fresh "
        "flow input after round-3's TI-based hidden interactions died."
    ),
    compute=compute,
)
