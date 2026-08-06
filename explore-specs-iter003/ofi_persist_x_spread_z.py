"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

ofi_persist_x_spread_z: fast book-flow sign-run commitment (recomputed
ofi_sign_persist_60s) x spread-state z -- stress-gated run structure.
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
    """ofi_sign_persist_60s base x z(quoted_spread_ticks, 300s)."""
    base = _ofi_persist()
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((base * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_persist_x_spread_z",
    mechanism=(
        "Stress-gated flow COMMITMENT measured by run structure, not "
        "magnitude: ofi_sign_persist_60s scores the fraction of "
        "consecutive ofi_15s readings keeping the same sign, signed by the "
        "current direction -- a magnitude-blind second-order statistic. "
        "Gating it by the spread-state z asks whether the COST of "
        "commitment is what makes it informative: under WIDE stressed "
        "spreads, keeping one-sided book flow running snapshot after "
        "snapshot bears full adverse-selection cost, so such commitment is "
        "deliberate informed accumulation -> continuation at 15-60s; under "
        "unusually TIGHT quotes the same sign runs are zero-cost churn "
        "(product flips them). Because the base ignores flow size "
        "entirely, this tests whether pure run STRUCTURE carries the "
        "stress-informed signal -- a different economic input than "
        "ofi_concord_x_spread_z (magnitude-capped agreement) and than "
        "ofi_z_x_spread_z (flow surprise)."
    ),
    info_set="ofi_15s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: ofi_sign_persist_60s (R3-C admitted) "
        "has no spread-z interaction yet; CKS (2014) OFI persistence "
        "interacted with the round-3 live gating dimension. Dedup note: "
        "the base sits on the round-3 watchlist (panel rho 0.782, rho "
        "0.723 vs ofi_concord_15_60); the gated forms decorrelate "
        "empirically (0.22-0.42 for round-3 products) but sibling corr "
        "with the ofi_concord gates needs screening."
    ),
    compute=compute,
)
