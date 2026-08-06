"""Explore-lane prototype spec (iter-003 R6D, family R6D).

oir_zaccel_ts_agree: timescale CONFIRMATION gate on oir z-acceleration.
Fast 15s z-acceleration (directional) amplified by the slow 60s
z-acceleration magnitude, but ONLY when both accelerations point the same
direction (confirmation); zero on disagreement. Weighted by |z|.
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
    z = _z(pl.col("oir"), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    d2z15 = dz15 - dz15.shift(LAG15)
    d2z60 = dz60 - dz60.shift(LAG60)
    agree = (d2z15.sign() == d2z60.sign()).cast(pl.Float64)
    return part.select((d2z15 * d2z60.abs() * z.abs() * agree).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zaccel_ts_agree",
    mechanism=(
        "Timescale-confirmation acceleration: the fast 15s z-acceleration "
        "of z_300(oir), amplified by the slow 60s z-acceleration magnitude, "
        "but emitted ONLY when both accelerations point the same direction. "
        "When the top-of-book imbalance regime is already stretched (high "
        "|z|) and BOTH the 15s and 60s curvature confirm each other "
        "(both building or both decaying), the consensus across timescales "
        "marks a mature, multi-horizon acceleration -- institutional "
        "quoting programs committing at increasing speed over both seconds "
        "and a full minute. The product form (fast acceleration x slow "
        "magnitude) ensures the signal scales with the STRENGTH of "
        "confirmation on both scales: a weak 60s acceleration barely "
        "amplifies the fast signal, while a strong one doubles it. Rows "
        "where the two timescales disagree are exactly zero: conflicting "
        "curvature is treated as noise (the slow and fast quoting "
        "processes are fighting, not confirming). Economically distinct "
        "from oir_zaccel_15s_x_60s: that carries the fast acceleration "
        "signed by the slow DIRECTION even on disagreement (betting the "
        "slow direction wins); this fires only on strict confirmation, "
        "making it sparser and more selective."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6D family brief direction 1: agreement-gated "
        "cross-timescale acceleration. The sign-match gate isolates "
        "high-confidence multi-horizon intensification events from "
        "single-timescale noise."
    ),
    compute=compute,
)
