"""Explore-lane prototype spec (iter-003 R6D, family R6D).

wdi_zaccel_ts_agree: timescale CONFIRMATION gate on wdi z-acceleration.
Fast 15s z-acceleration amplified by the slow 60s z-acceleration
magnitude, ONLY when both accelerations point the same direction; zero
on disagreement. Weighted by |z|.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG15 = 5
LAG60 = 20


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z15 * |d2z60| * |z| when signs agree; 0 when they disagree.

    Warm-up rows null (nulls propagate through the sign comparison).
    """
    z = _z(pl.col("wdi"), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    d2z15 = dz15 - dz15.shift(LAG15)
    d2z60 = dz60 - dz60.shift(LAG60)
    agree = (d2z15.sign() == d2z60.sign()).cast(pl.Float64)
    return part.select((d2z15 * d2z60.abs() * z.abs() * agree).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zaccel_ts_agree",
    mechanism=(
        "Timescale-confirmation acceleration: the fast 15s z-acceleration "
        "of z_300(wdi), amplified by the slow 60s z-acceleration magnitude, "
        "emitted ONLY when both accelerations agree in direction. When the "
        "5-level depth-imbalance regime is already stretched (high |z|) "
        "and BOTH the 15s and 60s curvature confirm (both intensifying or "
        "both decaying), the multi-horizon consensus marks a mature "
        "acceleration -- depth programs committing at increasing speed "
        "over both seconds and a full minute, which is costlier to fake "
        "than a single-scale push. The product form ensures the signal "
        "scales with confirmation STRENGTH: a weak 60s acceleration barely "
        "amplifies, a strong one doubles the fast signal. Disagreement "
        "rows are exactly zero: conflicting curvature across timescales is "
        "noise where the slow and fast quoting processes fight. Distinct "
        "from wdi_zaccel_15s_x_60s (which carries signal even on "
        "disagreement, betting the slow direction wins): this is a "
        "confirmation-only gate, sparser and more selective, zeroing out "
        "cross-timescale conflict entirely."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6D family brief direction 1: agreement-gated "
        "cross-timescale acceleration. The sign-match gate isolates "
        "high-confidence multi-horizon intensification from "
        "single-timescale noise."
    ),
    compute=compute,
)
