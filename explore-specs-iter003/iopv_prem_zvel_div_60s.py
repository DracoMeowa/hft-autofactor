"""Explore-lane prototype spec (iter-003 R5, family R5-A).

iopv_prem_zvel_div_60s: z-level vs instantaneous-velocity divergence on
iopv_premium, SIGNED-DIFFERENCE form -- the slow premium regime z minus
its own fast z-velocity, itself regime-normalized.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 20  # 20 x 3s rows = 60s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(iopv_premium, 300s) - z(dz, 300s) where dz = 60s z-velocity.

    Warm-up rows null: z warm-up propagates through dz into the
    velocity's own trailing z. Both terms regime-normalized.
    """
    z_e = _z(pl.col("iopv_premium"), W)
    dz_e = z_e - z_e.shift(LAG)
    tmp = part.select(z_e.alias("_z"), dz_e.alias("_dz"))
    tmp = tmp.select(pl.col("_z"), _z(pl.col("_dz"), W).alias("_dzz"))
    return tmp.select((pl.col("_z") - pl.col("_dzz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_prem_zvel_div_60s",
    mechanism=(
        "Premium overextension vs its own fast edge: z_300(iopv_premium) "
        "minus the trailing-300s z of its own 60s z-velocity. When the "
        "premium reads extreme but its velocity is already normalizing "
        "(large positive gap), the mispricing has peaked -- AP arbitrage "
        "has begun closing it, and the premium reverts toward IOPV at "
        "60-300s; when normalized velocity leads the level (large "
        "negative gap), the premium regime is still building and "
        "continues. The 60s window matches the AP response timescale. "
        "Both components are z-normalized against their own 300s "
        "distributions, making this a relative-stretch measure. DEDUP: "
        "library iopv_premium_z_120s/600s are pure LEVEL z (state only); "
        "library iopv_vel_drift_300s z-scores the iopv_VELOCITY engine "
        "column's drift (a different input); here the premium LEVEL z is "
        "tensioned against its OWN z-velocity, a different economic "
        "question: is the mispricing regime building or exhausting?"
    ),
    info_set="iopv_premium",
    inspiration=(
        "iter-003 R5-A family brief: signed-divergence form of the "
        "ofi_z_cross_vel_15s z-vs-velocity template applied to "
        "iopv_premium with a 60s velocity to match the AP arbitrage "
        "response timescale."
    ),
    compute=compute,
)
