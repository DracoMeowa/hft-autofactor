"""Explore-lane prototype spec (iter-003 R5, family R5-C).

oir_zcross_x_rvhi: touch-queue regime-flip velocity ISOLATED to the
high-volatility regime (rv_60s above its 300s rolling mean). Tests whether
touch-queue hand-offs carry long-horizon signal only during turbulence.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z / regime window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """oir crossing velocity, kept only when rv > rv_mean; else 0. warm-up null."""
    z = _z(pl.col("oir"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    cross_vel = flip * (z - z_lag)
    rv = pl.col("rv_60s")
    rv_mean = rv.rolling_mean(window_size=W, min_samples=W)
    gate = (
        pl.when(rv_mean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(rv > rv_mean)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
    )
    return part.select((cross_vel * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zcross_x_rvhi",
    mechanism=(
        "Turbulence-isolated touch-queue flips: oir_z_cross_vel_15s scores "
        "the cheapest-quotes regime hand-off -- the fastest, cheapest place "
        "to signal urgency, which informed traders touch first. But a touch "
        "flip in a CALM market is often routine touch rotation between "
        "competing market makers with no directional content. Hypothesis: "
        "the SAME touch hand-off during TURBULENCE (rv above its trailing "
        "mean) marks an abrupt informed queue takeover -- price is already "
        "moving, and seizing the touch queue is aggressive priority "
        "competition that continues in the new direction at 300-900s. The "
        "binary 0/1 gate zeroes all calm-regime flips (where the signal is "
        "noise) and retains only turbulent-regime flips, isolating the "
        "high-signal regime. Distinct from wdi_zvel_x_rv_regime (signed "
        "dual-regime on the CONTINUOUS velocity): this is a one-sided "
        "ISOLATION gate on a CROSSING EVENT -- it does not test the calm "
        "regime at all, only asks whether turbulence is the necessary "
        "condition for touch-flip signal."
    ),
    info_set="oir, rv_60s",
    inspiration=(
        "iter-003 R5-C family brief direction 2: condition z-vel winners "
        "on rv regime; this spec isolates the high-rv regime only "
        "(turbulence-amplifies-urgency hypothesis) on the oir crossing base."
    ),
    compute=compute,
)
