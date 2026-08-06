"""Explore-lane prototype spec (iter-003 R6A, family R6A).

microprice_dev_zaccel_extreme_15s: z-ACCELERATION-extremeness PRODUCT form
on microprice_dev. d2z * |z|. Mirrors the round-5 winning oir_zaccel
template on the queue-weighted fair-value lead. Distinct from the existing
microprice_dev_z_accel_sign_15s (z * sign(d2z), sign-only) and from the
round-4 velocity product (dz * |z|): this is the acceleration MAGNITUDE
weighted by level extremity.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback for velocity and acceleration


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| where d2z = 15s acceleration of the microprice-deviation z.

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(pl.col("microprice_dev"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="microprice_dev_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted fair-value-lead stretch: the 15s "
        "acceleration (2nd difference) of z_300(microprice_dev), weighted "
        "by how stretched that regime is (|z|). microprice_dev (micro "
        "minus mid, in px) is the queue-weighted fair-value lead -- its "
        "sign tells which side of the book carries the heavier queued "
        "interest. Its z-acceleration isolates INTENSIFYING queue-pressure "
        "divergence from steady-state tilt: when the fair-value lead is "
        "already stretched versus its 300s norm (high |z|) and its "
        "curvature is accelerating further, the queue imbalance is "
        "rebuilding at increasing speed -- informed repositioning that "
        "drags mid toward the microprice at 15-60s. Economically distinct "
        "from microprice_dev_z_accel_sign_15s (round-5, z * sign(d2z)): "
        "that discards magnitude and uses the binary acceleration "
        "direction as a gate; this RETAINS the acceleration magnitude and "
        "weights it by the level stretch. The product form fires on every "
        "row with nonzero curvature, scoring how hard the regime is "
        "curving scaled by how stretched it already is."
    ),
    info_set="microprice_dev",
    inspiration=(
        "iter-003 R6A family brief: z-acceleration-extremeness PRODUCT "
        "form on microprice_dev, filling the gap between the existing "
        "sign-only acceleration variant and the round-4 velocity product."
    ),
    compute=compute,
)
