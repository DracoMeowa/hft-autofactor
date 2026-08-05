"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

event_intens_z_300s: causal z-score of book_event_intensity_60s (feed
events per second) against its trailing 300s distribution -- an
information-arrival regime detector.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing reference window


def compute(part: pl.DataFrame) -> pl.Series:
    """Causal z-score of book_event_intensity_60s over 100 rows.

    Warm-up null; constant trailing windows (std == 0) map to 0.0.
    """
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
    name="event_intens_z_300s",
    mechanism=(
        "Feed-event intensity (trades and book-touching events per second) "
        "is a direct readout of the information-arrival rate, decoupled "
        "from both price moves and the 3s snapshot cadence. Information "
        "arrives in self-exciting clusters (Hawkes process): an intensity "
        "unusually high versus its own trailing-300s norm marks an "
        "episode ONSET, and self-excitation implies the cluster persists "
        "for minutes -- the episode has further to run. During such "
        "episodes the tape is dominated by information-driven flow, whose "
        "directional imprint keeps pushing price in the same direction at "
        "60-900s; conversely, unusually quiet tape (large negative z) is "
        "noise-dominated drift with no follow-through. The z framing "
        "removes the strong time-of-day seasonality of raw event counts "
        "(open/close heat), leaving regime-relative heat -- the meta-"
        "lesson-compliant transform of a slow state variable."
    ),
    info_set="book_event_intensity_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 4 (activity regime: z of "
        "book_event_intensity_60s). Hawkes self-excitation of order flow "
        "(Bacry, Mastromatteo & Muzy 2015); round-1 lesson that "
        "regime-relative transforms beat levels."
    ),
    compute=compute,
)
