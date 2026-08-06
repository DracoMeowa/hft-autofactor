"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

iopv_drift_wide_gate: sustained IOPV drift active ONLY under unusually wide
quoting (one-sided clip gate) -- drift that persists because stress slows
the arb that would erase it.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window (base and gate)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """iopv_vel_drift_300s base x clip(z(spread,300s), 0, inf)."""
    base = pl.col("iopv_velocity").rolling_mean(window_size=W, min_samples=W)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    gate = sp_z.clip(lower_bound=0.0)
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_drift_wide_gate",
    mechanism=(
        "Stress-PROTECTED fundamental drift, one-sided claim: the trailing-"
        "300s mean NAV velocity is scored only while the spread state is "
        "unusually WIDE, exactly zero otherwise. Hypothesis: a sustained "
        "anchor trend continues to drag the ETF at 300-900s precisely when "
        "stressed quoting slows the arbitrage that would otherwise close "
        "the tracking gap; drift episodes in comfortable regimes are "
        "arbitraged away continuously and carry no residual edge, so they "
        "are switched off (no fade claim -- the product form carries that "
        "stronger assertion). The construction isolates 'fundamental "
        "trend + slow arb' co-occurrence as an episode detector for the "
        "long horizons. Dedup note: nonzero rows are monotone in base x "
        "positive weight, so sibling corr with iopv_drift_x_spread_z may "
        "be material; regime selection is the distinct input."
    ),
    info_set="iopv_velocity, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: one-sided-gate variant for the "
        "iopv_vel_drift_300s base; slow arb-latency protection logic for "
        "300-900s horizons."
    ),
    compute=compute,
)
