"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

intensity_z_180s: regime-relative feed event intensity -- how fast the
order book is being rebuilt versus its own recent norm (activity/attention
regime, direction carried by concurrent flow via the interaction siblings).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(book_event_intensity_60s, 180s); constant windows -> 0.0."""
    x = pl.col("book_event_intensity_60s")
    mean = x.rolling_mean(window_size=W, min_samples=W)
    std = x.rolling_std(window_size=W, min_samples=W)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="intensity_z_180s",
    mechanism=(
        "Activity/attention regime: book_event_intensity_60s counts feed "
        "events per second; its trailing-180s z flags minutes when the "
        "book is being rebuilt unusually fast for this instrument-day. "
        "Elevated event intensity marks an information-arrival regime -- "
        "many participants revising quotes, pulling and re-placing, hitting "
        "the touch -- which for this ETF accompanies primary-market "
        "creation/redemption flows and informed re-pricing that show "
        "short-horizon drift persistence. It is direction-symmetric in "
        "isolation (an activity, not a direction), so its standalone rank "
        "signal comes from co-movement of activity bursts with concurrent "
        "one-sided flow; its main value is as the conditioning leg of the "
        "ofi/ti x intensity interactions, but the regime itself can carry "
        "predictive content where volatility/attention affects the "
        "magnitude and persistence of short-horizon returns."
    ),
    info_set="book_event_intensity_60s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 5 (intensity z); "
        "book_event_intensity_60s materialized 2026-08-06; attention/"
        "activity regimes (Barber & Odean 2008 attention-driven trading)."
    ),
    compute=compute,
)
