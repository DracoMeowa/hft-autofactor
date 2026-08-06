"""Explore-lane prototype spec (iter-003 R5, family R5-C).

oir_zcross_x_spread_z: touch-queue regime-flip velocity gated by spread-
state stress -- the admitted oir_z_cross_vel_15s base (z of oir crossed
zero within 15s; value is the crossing velocity) multiplied by the 300s
spread z. Tests whether touch-queue hand-offs predict longer-horizon
returns only when spreads are stressed.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """crossing velocity of oir * z(spread, 300s); warm-up null."""
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
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((cross_vel * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zcross_x_spread_z",
    mechanism=(
        "Spread-stress-gated touch-queue regime flips: oir_z_cross_vel_15s "
        "(the admitted crossing template on best-bid/best-ask imbalance) "
        "scores sign-reversal events of the touch regime -- moments the "
        "touch queue changes hands -- but these flips are a mix of informed "
        "hand-offs and routine touch rotation. Hypothesis: a touch-queue "
        "hand-off under WIDE, stressed spreads is far more credible as "
        "informed: pulling and refilling at the best quotes is costly when "
        "the spread is wide (the adverse-selection tax is high), so only "
        "informed traders pay it; under tight, comfortable spreads the same "
        "flip is cheap routine competition. Multiplying the crossing "
        "velocity by z(spread, 300s) retains the crossing event-sparse "
        "structure but re-weights each flip by the contemporaneous spread "
        "stress, isolating the informed hand-offs whose direction survives "
        "at 300-900s. Distinct from ofi_z_x_spread_z (round-1: z-SURPRISE "
        "of single-window OFI x spread, continuous) -- here the gated input "
        "is a CROSSING EVENT (event-sparse, regime-transition not surprise)."
    ),
    info_set="oir, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-C family brief: condition the round-4 z-velocity "
        "winner oir_z_cross_vel_15s on spread-z to extend its horizon; "
        "spread-z is the only consistently live interaction dimension."
    ),
    compute=compute,
)
