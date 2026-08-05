"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

buysell_regime_gap_300s: which aggressive SIDE is more stretched against
its OWN recent regime -- z_300(buy_vol_60s) minus z_300(sell_vol_60s).
Regime-adjusted side dominance, not the contemporaneous imbalance level.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window per side


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(buy_vol_60s, 300s) - z(sell_vol_60s, 300s); warm-up rows null."""
    zb = _z(pl.col("buy_vol_60s"), W)
    zs = _z(pl.col("sell_vol_60s"), W)
    return part.select((zb - zs).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="buysell_regime_gap_300s",
    mechanism=(
        "Side-specific regime stretch: the naive buy/sell asymmetry "
        "(B-S)/(B+S) is, by construction, the panel's "
        "trade_imbalance_60s -- a dedup death sentence. But asking "
        "WHICH SIDE's volume is more elevated against ITS OWN trailing-"
        "300s regime is a genuinely different question. When both sides "
        "swell together (a two-sided volume burst) the two z-scores "
        "cancel and the factor stays near 0 -- churn/attention, no "
        "direction. When only the buy side is stretched against its own "
        "recent history while the sell side is normal-or-quiet, the "
        "tape is in one-sided accumulation; the mirror is distribution. "
        "One-sided regime stretch marks a committed directional "
        "participant whose schedule continues, predicting 60-900s drift "
        "in the stretched side's direction. The per-side z-referencing "
        "removes the shared volume-level component that makes raw "
        "asymmetry a clone of trade imbalance, so rank correlation with "
        "panel trade_imbalance_60s is bounded well below the dedup "
        "wall by construction."
    ),
    info_set="buy_vol_60s, sell_vol_60s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R3-C brief direction 4 (buy/sell vol structure: "
        "asymmetry and its short-window z); round-2 lesson that raw "
        "fast columns must be regime-normalized to survive the panel "
        "wall; per-side z-referencing as the decorrelation lever that "
        "keeps this away from trade_imbalance_60s."
    ),
    compute=compute,
)
