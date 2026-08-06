"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

iopv_drift_x_spread_z: sustained 300s IOPV drift (recomputed
iopv_vel_drift_300s) x spread-state z -- slow fundamental regime gated by
slow quoting stress.
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
    """iopv_vel_drift_300s base x z(quoted_spread_ticks, 300s)."""
    base = pl.col("iopv_velocity").rolling_mean(window_size=W, min_samples=W)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((base * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_drift_x_spread_z",
    mechanism=(
        "Slow-regime x slow-regime: iopv_vel_drift_300s measures whether "
        "the NAV anchor has been PERSISTENTLY trending (sustained "
        "fundamental drift, not a flicker). Under concurrent quoting "
        "STRESS (spread-state z high), market makers are reluctant to "
        "refresh quotes against a moving anchor and arbitrage is costly, "
        "so the ETF lags the drifting basket for many minutes -> "
        "continuation of the drift direction at 300-900s, the horizons "
        "where the base itself lived (round-2 900s pass). Under "
        "comfortably tight quoting the same sustained drift gets worked "
        "off continuously by cheap arbitrage, leaving little residual "
        "edge (product flips it). This is the only R4-A pairing that "
        "interacts two SLOW 300s states, aimed at the long horizons where "
        "accumulation/regime state carried 4/6 of the eval-v2 re-screen "
        "passes; it is economically distinct from iopv_velz_x_spread_z "
        "(transient shock x stress), which asks about spike ONSETS."
    ),
    info_set="iopv_velocity, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: iopv_vel_drift_300s (R2-D admitted, "
        "900s alive) has no spread-z interaction; round-1 meta-lesson that "
        "300-900s horizons reward regime state + round-3 that spread-z is "
        "the only live interaction dimension."
    ),
    compute=compute,
)
