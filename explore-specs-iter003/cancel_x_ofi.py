"""Explore-lane prototype spec (iter-003, flow-interaction lens).

cancel_x_ofi: cancellation intensity gated by the direction of book flow --
quotes being pulled in step with (or ahead of) directional book building.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s cancel-intensity z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(cancel_ratio_60s, 180s) x sign(ofi_60s)."""
    cx_z = _z(pl.col("cancel_ratio_60s"), W)
    direction = pl.col("ofi_60s").sign()
    return part.select((cx_z * direction).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="cancel_x_ofi",
    mechanism=(
        "Quote-pull signature: cancel_ratio measures how much of message "
        "traffic is cancellations -- high values mean liquidity is being "
        "placed and yanked quickly. Direction-free, it is noise (HFT "
        "churn is constant background). Conditioned on the sign of book "
        "flow it becomes a behavior fingerprint: heavy cancelling while "
        "the book builds bid-side = market makers/algms pulling offers "
        "ahead of an expected up-move (or spoof-layering on the bid), "
        "both of which precede continuation in the OFI direction at "
        "15-60s. Cancellation intensity is the message-traffic channel "
        "no other prototype reads."
    ),
    info_set="cancel_ratio_60s, ofi_60s (library)",
    inspiration=(
        "iter-003 family brief seed 11; cancel-to-trade message dynamics "
        "as HFT footprint (Cont, Kukanov & Stoikov 2014 on order-flow "
        "composition); signed/conditioned repair per the iter-002 "
        "meta-lesson."
    ),
    compute=compute,
)
