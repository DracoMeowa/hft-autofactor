"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

ti_accel_x_spread_z: aggressive-flow acceleration (recomputed ti_accel_15_60
= trade_imbalance_15s - trade_imbalance_60s) x spread-state z -- urgency
gated by quoting stress.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing spread-state window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """ti_accel_15_60 base x z(quoted_spread_ticks, 300s); warm-up null."""
    base = pl.col("trade_imbalance_15s") - pl.col("trade_imbalance_60s")
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((base * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_accel_x_spread_z",
    mechanism=(
        "Stress-gated taker urgency: ti_accel_15_60 contrasts the freshest "
        "15s of signed aggressive volume with the trailing minute -- "
        "marketable flow intensifying RIGHT NOW. When that acceleration "
        "happens while quoted spreads are unusually WIDE, the aggressor is "
        "sweeping a thin, fearful stack: makers have already stepped back "
        "(wide quotes = adverse-selection fear), yet someone is still "
        "hitting/lifting with increasing urgency -- the classic informed-"
        "sweeping state, and the consumed resting liquidity propagates the "
        "move in the aggression direction at 15-60s. The same acceleration "
        "under unusually TIGHT comfortable quoting is benign retail/ "
        "competition noise against a deep, relaxed stack, whose direction "
        "carries little (product flips it). This gates the ACTIVE trade "
        "channel by the quoting regime -- distinct from ofi_z_x_spread_z "
        "(passive book-flow surprise) and from the ungated base."
    ),
    info_set="trade_imbalance_15s, trade_imbalance_60s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: ti_accel_15_60 (R2-B admitted) still "
        "lacks a spread-z interaction; round-3 established spread-z as the "
        "only live interaction dimension; urgency-under-fear logic mirrors "
        "the div_z_x_spread_z stress-gating mechanism on the trade channel."
    ),
    compute=compute,
)
