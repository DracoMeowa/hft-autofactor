"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

gap_slope_thinwalk_300s: interaction -- regime-adjusted aggressor gap x
NEGATIVE regime-adjusted book thickness. Aggression walking a thin book.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment (588000)
W = 100       # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(gap_ticks, 300s) x (-z(book_slope, 300s)); warm-up null."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    gap_z = _z(gap_ticks, W)
    slope_z = _z(pl.col("book_slope"), W)
    return part.select((gap_z * (-slope_z)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="gap_slope_thinwalk_300s",
    mechanism=(
        "Aggression amplified by a thin book: the last-mid gap marks an "
        "aggressor crossing the spread; book_slope marks how much "
        "inventory wall stands behind the touch to stop the walk. An "
        "aggressive print (large gap z) landing while the depth profile is "
        "UNUSUALLY FLAT (low slope z -- thin walls) has little cushion "
        "behind the quote, so it tends to keep walking the book "
        "(continuation in the gap direction, larger impact). The same "
        "print landing into an unusually STEEP/thick book is absorbed by "
        "stacked depth and reverts. Multiplying gap-z by NEGATIVE "
        "thickness-z is large exactly in the thin-book-aggression cell and "
        "signed by the aggressor direction, so a positive value = upward "
        "print into a thin book -> up continuation expected. This is the "
        "cross-column quote-shape pair (aggressor gap x book shape) and "
        "uses the gap as the pressure leg, distinguishing it from "
        "micro_slope_unsupported_300s (microprice leg)."
    ),
    info_set="last_px, mid_px, book_slope",
    inspiration=(
        "iter-003 R4-D brief direction (d) cross-column quote-shape; "
        "queue-walking / thin-book impact amplification (Lillo, Farmer & "
        "Mantegna 2003); both legs z-normalized per the ratio/z rule."
    ),
    compute=compute,
)
