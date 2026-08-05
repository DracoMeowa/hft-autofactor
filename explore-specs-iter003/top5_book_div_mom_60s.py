"""Explore-lane prototype spec (iter-003 R2-C, fullbook-depth lens).

top5_book_div_mom_60s: 60s momentum of the DIVERGENCE between the
visible 5-level imbalance (wdi) and the full-book imbalance -- touch
leading vs lagging the broad queue.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s momentum window


def _divergence() -> pl.Expr:
    """wdi - full-book imbalance; both bounded in [-1, 1]."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return pl.col("wdi") - fbi


def compute(part: pl.DataFrame) -> pl.Series:
    """60s delta of (wdi - full-book imbalance); warm-up rows null."""
    return part.select(_divergence().diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="top5_book_div_mom_60s",
    mechanism=(
        "Touch-vs-book divergence momentum: wdi measures the visible "
        "5-level imbalance while the full-book ratio measures the whole "
        "queue, so their difference isolates whether the executable tip "
        "is leading or lagging the broad queue. The 60s CHANGE of this "
        "gap catches the tip breaking away in real time: visible bid "
        "pressure rising FASTER than the whole book's means urgency is "
        "concentrating at the touch ahead of the deep queue (front-loaded "
        "demand that consumes the ask stack within 15-60s, upward "
        "continuation); the mirror -- broad book improving while the "
        "touch lags -- flags support still on its way to the executable "
        "levels. This lead-lag misalignment between queue layers is "
        "invisible to either imbalance alone and is a different economic "
        "question than either one's own momentum."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R2-C family brief direction 4 (five-level vs full-book "
        "sign/magnitude divergence); touch-vs-queue lead-lag structure; "
        "batch-2 total_*_vol materialized 2026-08-06; round-1 lesson: "
        "fast book-state deltas carry 15-60s."
    ),
    compute=compute,
)
