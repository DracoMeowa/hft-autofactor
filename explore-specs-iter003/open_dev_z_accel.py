"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_dev_z_accel: 60s DIFF of the 300s regime-z of the open-deviation --
second derivative of the stretch: is overextension building faster or
fading vs the day's own deviation regime?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_Z = 100  # 100 x 3s rows = 300s z window
W_D = 20   # 20 x 3s rows = 60s acceleration window


def compute(part: pl.DataFrame) -> pl.Series:
    """z_dev(300s)(i) - z_dev(300s)(i-20); warm-up rows null."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    m = dev.rolling_mean(window_size=W_Z, min_samples=W_Z)
    s = dev.rolling_std(window_size=W_Z, min_samples=W_Z)
    z = pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(
        (dev - m) / s
    )
    return part.select(z.diff(W_D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="open_dev_z_accel",
    mechanism=(
        "Second derivative of the anchor stretch: the round-2 open-"
        "deviation level says HOW FAR, its rolling z says HOW UNUSUAL, "
        "this asks WHETHER THE UNUSUALNESS IS STILL BUILDING. A rising z "
        "(positive value) = the stretch is accelerating relative to the "
        "morning's own regime: fresh one-sided pressure is overcoming the "
        "anchor pull RIGHT NOW -- either informed continuation or the "
        "last gasp before exhaustion, and the two have opposite short-"
        "horizon consequences that the IC sign resolves empirically. A "
        "falling z = the overextension is already decaying: reversion has "
        "started in relative terms even if price is still far from the "
        "open, and decaying overextension predicts continued drift back "
        "toward the anchor. Differencing the z (not the price) removes "
        "the momentum content: this is change in regime-surprise, a "
        "distinct quantity from price acceleration (which compares two "
        "raw momentum windows)."
    ),
    info_set="mid_px, open_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 1 (second "
        "derivative: stretch accelerating vs fading); derivative-of-"
        "regime-state pattern per the round-1 meta-lesson (deltas and "
        "accelerations carry signal where levels die)."
    ),
    compute=compute,
)
