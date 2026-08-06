"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_flow_align_net: signed NET of the two hidden-flow confirmations --
(bid-support z x buy-vol z) minus (ask-supply z x sell-vol z): which
hidden-flow regime currently dominates.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window on all four parents


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
    """(hb_z x buy_z) - (ha_z x sell_z); warm-up rows null."""
    hb_z = _z(_hidden_bid_share(), W)
    ha_z = _z(_hidden_ask_share(), W)
    buy_z = _z(pl.col("buy_vol_60s").cast(pl.Float64), W)
    sell_z = _z(pl.col("sell_vol_60s").cast(pl.Float64), W)
    net = hb_z * buy_z - ha_z * sell_z
    return part.select(net.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_flow_align_net",
    mechanism=(
        "Which hidden-flow regime dominates: bid-side confirmation (hidden "
        "bid support z x buy-volume z) MINUS ask-side confirmation (hidden "
        "ask supply z x sell-volume z). Positive = the demand regime "
        "(patient bid reservoirs + buy aggression) is stronger than the "
        "supply regime -> net hidden-backed upward pressure at 15-60s; "
        "negative = supply regime dominates. The difference is the "
        "economically relevant object: when BOTH sides are deep and BOTH "
        "gross volumes are heavy (a general high-liquidity/high-activity "
        "state), the two products common-mode cancel and the factor stays "
        "near zero -- it isolates the DIRECTIONAL asymmetry of hidden-"
        "backed flow, which a single-side product cannot. This is a signed "
        "combination of two fresh z-products over side-attributed volumes; "
        "round-3's TI-based hidden interactions (net imbalance operands) "
        "died, and no admitted library factor combines hidden-side regimes "
        "with side volumes in either form."
    ),
    info_set=(
        "total_bid_vol, total_ask_vol, depth_bid5, depth_ask5, "
        "buy_vol_60s, sell_vol_60s (batch-2)"
    ),
    inspiration=(
        "iter-003 R4-B brief direction (c): signed net of the hidden-flow "
        "alignment products; common-mode cancellation of the two-sided "
        "activity state as the identifying restriction."
    ),
    compute=compute,
)
