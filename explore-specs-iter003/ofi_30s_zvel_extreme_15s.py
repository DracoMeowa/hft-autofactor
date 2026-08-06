"""Explore-lane prototype spec (iter-003 R6D, family R6D).

ofi_30s_zvel_extreme_15s: extremity-weighted z-VELOCITY product on
ofi_30s (medium-window order-flow imbalance). The 15s change rate of
z_300(ofi_30s) weighted by |z|. Tests the zvel-extreme template on a
substrate between the fast ofi_15s and the slow ofi_60s.
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
    """dz * |z| where dz = 15s z-velocity of ofi_30s; warm-up null."""
    z = _z(pl.col("ofi_30s"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_30s_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted medium-window book-flow velocity: the 15s "
        "change rate of z_300(ofi_30s), weighted by how extreme the flow "
        "regime is (|z|). ofi_30s averages order-book-delta flow over "
        "half a minute -- between the burst-resolution of ofi_15s and "
        "the minute-norm of ofi_60s. This intermediate smoothing captures "
        "flow that persists across several snapshots but is not yet the "
        "minute average. When the 30s flow regime is already stretched "
        "(high |z|: persistent one-sided book pressure beyond the 300s "
        "norm) and its z is moving fast (large dz), the half-minute "
        "book-building program is being rapidly escalated -- informed "
        "passive flow that sustained for 30s and is now intensifying, "
        "whose impact continues at 15-60s. The |z| weight suppresses "
        "velocity around neutral regimes (routine churn). Economically "
        "distinct from ofi_15s_zvel_extreme_15s (faster substrate, "
        "captures single-snapshot bursts) and ofi_60s_zvel_extreme_15s "
        "(slower substrate, captures minute-scale escalation): the 30s "
        "window isolates multi-snapshot persistent flow dynamics that "
        "neither the 15s nor the 60s can resolve."
    ),
    info_set="ofi_30s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel velocity substrate. "
        "ofi_30s confirmed on the panel (used in ofi_accel_z_180s) but "
        "has not been put through the zvel-extreme template. The 30s "
        "smoothing is a distinct resolution scale."
    ),
    compute=compute,
)
