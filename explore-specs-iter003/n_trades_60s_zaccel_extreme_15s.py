"""Explore-lane prototype spec (iter-003 R6D, family R6D).

n_trades_60s_zaccel_extreme_15s: extremity-weighted z-ACCELERATION
product on the trade-arrival rate (n_trades_60s). The 15s acceleration
(2nd difference) of z_300(n_trades_60s) weighted by |z|. Tests whether
the CURVATURE of trading-intensity intensification carries signal.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG = 5


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| where d2z = 15s z-acceleration of n_trades_60s; warm-up null."""
    z = _z(pl.col("n_trades_60s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="n_trades_60s_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted trade-arrival regime stretch: the 15s "
        "acceleration of z_300(n_trades_60s), weighted by how abnormal "
        "the arrival regime is (|z|). n_trades_60s counts trailing-minute "
        "executions. Its z-acceleration isolates INTENSIFYING trading "
        "urgency from steady-state: when the arrival regime is already "
        "extreme (high |z|: a burst regime far above the 300s norm) and "
        "its rate of change is ITSELF accelerating, the surge of "
        "aggressive order flow is ramping up at increasing speed -- "
        "participants are hitting quotes with escalating urgency, which "
        "is a stronger continuation signal than a steady burst (constant "
        "velocity, zero acceleration). When d2z is negative and |z| is "
        "high, the burst's rate of intensification is decelerating -- "
        "the urgency is fading and impact tends to revert. The |z| weight "
        "ensures only already-abnormal arrival regimes contribute. "
        "Economically distinct from n_trades_60s_zvel_extreme_15s "
        "(velocity x |z|): that measures steady fast change of the "
        "arrival rate; this measures whether the change rate is itself "
        "increasing or decreasing -- the curvature of urgency. Also "
        "distinct from the dead trade_count_z_300s / ntrades_pace_z_300s "
        "(LEVEL statistics): the acceleration-extremity product is a "
        "derivative measure, not a level."
    ),
    info_set="n_trades_60s (wishlist batch-1)",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel z-acceleration "
        "substrate. The zaccel-extreme template was round-5's strongest "
        "construction; n_trades_60s is confirmed on the panel and has not "
        "been put through this template."
    ),
    compute=compute,
)
