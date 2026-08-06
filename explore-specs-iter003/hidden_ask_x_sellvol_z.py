"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_ask_x_sellvol_z: hidden ask-supply z x z(sell_vol_60s) -- patient ask
reservoirs ALIGNED with aggressive selling (supply-regime confirmation), both
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


def _hidden_ask_share() -> pl.Expr:
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    return (
        pl.when(ta > 0.0)
        .then(ha / ta)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ask hidden share, 300s) x z(sell_vol_60s, 300s); warm-up null."""
    ha_z = _z(_hidden_ask_share(), W)
    sell_z = _z(pl.col("sell_vol_60s").cast(pl.Float64), W)
    return part.select((ha_z * sell_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_ask_x_sellvol_z",
    mechanism=(
        "Supply-regime confirmation across channels: the product of the "
        "ask-side hidden-supply z and the z of SIDE-ATTRIBUTED sell volume. "
        "When both are high, aggressive selling is happening WHILE an "
        "unusually deep patient-ask overhang sits above the touch -- the "
        "visible sell aggression is backed by queued distribution that "
        "keeps refilling the offer, so downside pressure persists and "
        "rallies are capped at 15-60s. When sell flow runs against a "
        "SHALLOW hidden overhang, the selling has no reserve behind it and "
        "is more likely to exhaust/rebound. Supply-side twin of the bid "
        "confirmation, not its sign-flip: the ask-side hidden share and "
        "sell volume carry independent information (both sides can be "
        "simultaneously confirmed in heavy two-sided trading). The round-3 "
        "hidden-x-flow attempts used NET trade imbalance and died; the "
        "input here is one-sided GROSS sell volume, which remains high when "
        "net TI nets out to zero -- a regime the dead TI-products are "
        "structurally blind to. Both parents z-scored."
    ),
    info_set="total_ask_vol, depth_ask5, sell_vol_60s (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (c): hidden_ask_supply x sell_vol_60s "
        "alignment in z-form; side-attributed batch-2 volumes as the fresh "
        "flow input after round-3's TI-based hidden interactions died."
    ),
    compute=compute,
)
