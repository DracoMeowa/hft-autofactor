"""Explore-lane prototype spec (iter-003 R6D, family R6D).

ofi_30s_zaccel_extreme_15s: extremity-weighted z-ACCELERATION product on
ofi_30s. The 15s acceleration (2nd difference) of z_300(ofi_30s) weighted
by regime extremity |z|. Tests curvature intensification on the
intermediate OFI window.
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
    """d2z * |z| where d2z = 15s z-acceleration of ofi_30s; warm-up null."""
    z = _z(pl.col("ofi_30s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_30s_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted medium-window book-flow regime stretch: "
        "the 15s acceleration of z_300(ofi_30s), weighted by how stretched "
        "the regime is (|z|). ofi_30s averages order-book-delta flow over "
        "half a minute -- a resolution between burst-level (15s) and "
        "minute-norm (60s). Its z-acceleration isolates INTENSIFYING "
        "multi-snapshot book-building from steady-state: when the 30s "
        "flow regime is already extreme (high |z|) and its rate of change "
        "is accelerating, the half-minute passive program is ramping up "
        "-- persistent flow that was building steadily for 30s and is now "
        "escalating further, a stronger commitment signal than either a "
        "15s flicker or a 60s slow drift. The direction of acceleration "
        "continues at 15-60s. Economically distinct from "
        "ofi_15s_zaccel_extreme_15s (burst curvature) and "
        "ofi_60s_zaccel_extreme_15s (minute curvature): the 30s window "
        "isolates the multi-snapshot persistence scale where informed "
        "flow programs operate."
    ),
    info_set="ofi_30s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel z-acceleration "
        "substrate. ofi_30s is an underused panel column (appears only "
        "in ofi_accel_z_180s); the zaccel-extreme template has not been "
        "applied to this intermediate window."
    ),
    compute=compute,
)
