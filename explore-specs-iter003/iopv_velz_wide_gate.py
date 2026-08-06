"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

iopv_velz_wide_gate: IOPV velocity surprise active ONLY under unusually wide
quoting (one-sided clip gate) -- stress-trapped arbitrage shocks.
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
    """iopv_vel_z_300s base x clip(z(spread,300s), 0, inf); warm-up null."""
    base = _z(pl.col("iopv_velocity"), W)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    gate = sp_z.clip(lower_bound=0.0)
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_velz_wide_gate",
    mechanism=(
        "Stress-TRAPPED arbitrage shocks, one-sided claim: the IOPV-"
        "velocity z surprise is scored only while the spread state is "
        "unusually WIDE, exactly zero otherwise. Hypothesis: NAV-velocity "
        "shocks continue at 15-300s horizons only when quoting stress "
        "makes the closing arbitrage expensive, so the tracking gap stays "
        "open; shocks arriving in comfortable regimes are arbitraged away "
        "too fast to be tradeable and are switched off with no mean-"
        "reversion claim (the product form additionally bets on tight-"
        "regime snap-back; this spec does not). The construction is an "
        "episode detector for 'fundamental move + trapped arb', the subset "
        "of velocity spikes with the slowest decay. Dedup note: nonzero "
        "rows are monotone in base x positive weight, so sibling corr "
        "with iopv_velz_x_spread_z may be material; the regime selection "
        "(zero mass outside stress) is the distinct input."
    ),
    info_set="iopv_velocity, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: one-sided-gate variant for the "
        "iopv_vel_z_300s base; arbitrage-latency logic restricted to the "
        "stress subset where the latency is longest."
    ),
    compute=compute,
)
