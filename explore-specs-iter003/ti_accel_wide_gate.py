"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

ti_accel_wide_gate: aggressive-flow acceleration active ONLY under unusually
wide quoting (one-sided clip gate) -- urgency that matters only in stress.
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
    """ti_accel_15_60 base x clip(z(spread,300s), 0, inf); warm-up null."""
    base = pl.col("trade_imbalance_15s") - pl.col("trade_imbalance_60s")
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    gate = sp_z.clip(lower_bound=0.0)
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_accel_wide_gate",
    mechanism=(
        "Stress-UNLOCKED taker urgency, one-sided claim: the fast-minus-"
        "slow trade-imbalance gap is scored only while the spread state is "
        "unusually WIDE, exactly zero otherwise. Hypothesis: aggressive-"
        "flow acceleration is informative exclusively in stressed quoting "
        "regimes, where makers have already widened in fear and fresh "
        "urgency against that withdrawn liquidity marks informed sweeping "
        "-> continuation of the aggression direction at 15-60s; in normal/"
        "tight regimes the same acceleration is routine noise and is "
        "switched off with no fade claim (the product form additionally "
        "bets against tight-regime acceleration; this spec does not). The "
        "exact-zero mass outside stress rows makes the cross-section an "
        "episode detector rather than a continuously signed signal. Dedup "
        "note: nonzero rows are monotone in base x positive weight, so "
        "sibling corr with ti_accel_x_spread_z may be material; the "
        "distinct economic input is the regime SELECTION."
    ),
    info_set="trade_imbalance_15s, trade_imbalance_60s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: one-sided-gate variant for the "
        "ti_accel_15_60 base (R2-B admitted, no spread interaction yet); "
        "episode-detector construction: only stressed minutes count."
    ),
    compute=compute,
)
