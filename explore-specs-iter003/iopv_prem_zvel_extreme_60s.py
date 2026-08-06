"""Explore-lane prototype spec (iter-003 R5, family R5-A).

iopv_prem_zvel_extreme_60s: z-level vs instantaneous-velocity divergence on
iopv_premium, PRODUCT form -- the 60s z-velocity of the premium regime
weighted by the extremity |z| of the regime being moved.
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
    """dz * |z| where dz = z_now - z_60s_ago; warm-up rows null."""
    z = _z(pl.col("iopv_premium"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_prem_zvel_extreme_60s",
    mechanism=(
        "Extremity-weighted premium-regime velocity: the 60s change rate "
        "of z_300(iopv_premium), weighted by how extreme the regime "
        "being moved is (|z|). The ETF premium versus IOPV measures "
        "transient mispricing: when the premium is already stretched "
        "(high |z|, far from its recent 300s norm) and still accelerating "
        "in the same direction over 60s, the mispricing is widening "
        "beyond what AP arbitrage has absorbed -- informed flow is "
        "pushing the ETF ahead of its basket faster than redemption "
        "mechanics can respond. The 60s window captures the AP response "
        "timescale (creation/redemption is not instantaneous), so "
        "decisive motion of a stretched premium predicts the direction "
        "of continued pressure at 60-300s before the arb closes. The "
        "extremity weight zeroes out routine premium jitter around the "
        "norm. DEDUP: library iopv_premium_z_120s and z_600s are LEVEL z "
        "(state only, no velocity); library iopv_vel_z_300s z-scores the "
        "iopv_velocity engine column (a different input); library "
        "iopv_premium_mom_60s is a raw unnormalized 60s delta that died "
        "in iter-001. This is the only premium factor combining the "
        "level extremity with the level's OWN 60s velocity in "
        "regime-normalized form."
    ),
    info_set="iopv_premium",
    inspiration=(
        "iter-003 R5-A family brief: product (extreme) form of the "
        "ofi_z_cross_vel_15s z-vs-velocity template applied to "
        "iopv_premium with a 60s velocity to match the AP response "
        "timescale; round-4 showed the extreme construction was the "
        "goldmine (wdi_zvel_extreme_15s hit 15s IC +0.18)."
    ),
    compute=compute,
)
