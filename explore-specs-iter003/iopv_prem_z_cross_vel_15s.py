"""Explore-lane prototype spec (iter-003 R5, family R5-A).

iopv_prem_z_cross_vel_15s: z-level vs instantaneous-velocity divergence on
iopv_premium, CROSSING form -- the 300s z of the ETF premium crossed zero
within the last 15s; value is the z-velocity, only at crossings, else 0.
Premium-arbitrage pressure reversal events.
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
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the premium-regime reversal.
    """
    z = _z(pl.col("iopv_premium"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((flip * (z - z_lag)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_prem_z_cross_vel_15s",
    mechanism=(
        "Premium-arbitrage regime reversal events: the trailing-300s z of "
        "iopv_premium (ETF price vs IOPV, ratio) crosses zero within 15s. "
        "The premium has a structural component (fund-flow regimes "
        "persisting hours) and a transient component (flow bursts, stale "
        "IOPV after fast basket moves). A zero-crossing of the 300s z "
        "means the transient component just flipped from above-norm "
        "(overpriced vs fair value) to below-norm (underpriced) or vice "
        "versa -- the direction of AP (authorized participant) "
        "creation/redemption pressure has reversed. The crossing VELOCITY "
        "measures how fast the reversal is engaging: a decisive fast "
        "crossing marks an AP program committing capital to the new "
        "direction, whose impact continues at 15-60s while the arb "
        "executes; a slow drift across zero scores weak. Event-sparse (0 "
        "off crossings). DEDUP: library iopv_premium_z_120s (and z_600s) "
        "are pure LEVEL z active every row (regime STATE); here only the "
        "transition EVENT is scored. Library iopv_vel_z_300s and "
        "iopv_vel_drift_300s z the IOPV VELOCITY column (a different "
        "input); here the premium LEVEL itself is z-scored and crossed "
        "with its own velocity. Round-1 lesson: unconditional premium "
        "levels dead (0/8); DYNAMICS are the live form -- the crossing "
        "template extends that finding from velocity-levels to "
        "level-reversal events."
    ),
    info_set="iopv_premium",
    inspiration=(
        "iter-003 R5-A family brief: apply the admitted "
        "ofi_z_cross_vel_15s z-vs-velocity crossing template to the "
        "iopv_premium state column (never given z-vel treatment); "
        "round-1/2 lesson that premium dynamics live while levels die."
    ),
    compute=compute,
)
