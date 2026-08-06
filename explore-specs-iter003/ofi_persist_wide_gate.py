"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

ofi_persist_wide_gate: fast book-flow sign-run commitment active ONLY under
unusually wide quoting (one-sided clip gate).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing spread-state window
B = 20   # 20 x 3s rows = 60s trailing persistence window (as in the base)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _ofi_persist() -> pl.Expr:
    """Recomputed ofi_sign_persist_60s: (same-sign pair share - 0.5)*2*sign."""
    sgn = pl.col("ofi_15s").sign()
    sgn_lag = sgn.shift(1)
    same = (
        pl.when(sgn.is_null() | sgn_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((sgn == sgn_lag) & (sgn != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    share = same.rolling_mean(window_size=B, min_samples=B)
    return (share - 0.5) * 2.0 * sgn


def compute(part: pl.DataFrame) -> pl.Series:
    """ofi_sign_persist_60s base x clip(z(spread,300s), 0, inf)."""
    base = _ofi_persist()
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    gate = sp_z.clip(lower_bound=0.0)
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_persist_wide_gate",
    mechanism=(
        "Stress-UNLOCKED flow commitment, one-sided claim: the sign-run "
        "persistence of ofi_15s is scored only while the spread state is "
        "unusually WIDE vs its trailing 300s, exactly zero otherwise. "
        "Hypothesis: committed one-sided flow runs are informative ONLY in "
        "stressed regimes, where sustaining them bears adverse-selection "
        "cost and therefore reveals deliberate positioning -> continuation "
        "at 15-60s; in comfortable regimes runs are costless churn and are "
        "switched off (no fade claim -- the product form carries that "
        "stronger assertion). As a magnitude-blind run statistic gated to "
        "stress episodes, this is an episode detector for informed "
        "accumulation that is structurally distinct from the ungated base "
        "(mass of exact zeros outside stress). Dedup note: nonzero rows "
        "are monotone in base x positive weight; sibling corr with "
        "ofi_persist_x_spread_z may be material, and the watchlist "
        "proximity to the ofi_concord gates applies."
    ),
    info_set="ofi_15s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: one-sided-gate variant for the "
        "ofi_sign_persist_60s base (R3-C admitted, no spread interaction "
        "yet); commitment-cost logic under stressed quoting."
    ),
    compute=compute,
)
