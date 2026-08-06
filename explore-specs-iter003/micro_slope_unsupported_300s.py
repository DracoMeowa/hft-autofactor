"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

micro_slope_unsupported_300s: microprice price pressure WITHOUT book-shape
support -- pressure in a thin book vs pressure backed by a thick book.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(microprice_dev, 300s) x (-z(book_slope, 300s)); warm-up null."""
    micro_z = _z(pl.col("microprice_dev"), W)
    slope_z = _z(pl.col("book_slope"), W)
    return part.select((micro_z * (-slope_z)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="micro_slope_unsupported_300s",
    mechanism=(
        "Pressure-without-support across two quote-shape channels: "
        "book_slope is the mean OLS slope of ln(cumulative depth) against "
        "distance-from-mid over both sides -- a high slope means depth "
        "accumulates fast away from the touch (a thick, well-stocked book), "
        "a low slope means depth builds slowly (thin walls behind the "
        "touch). Microprice pressure (microprice_dev z) that arrives while "
        "the book is UNUSUALLY THIN (low slope z) has little visible "
        "inventory behind the touch to absorb it: the same imbalance moves "
        "price further, so unsupported pressure tends to CONVERT into mid "
        "drift in the pressure direction (continuation). The identical "
        "pressure arriving into an unusually thick book is backed/absorbed "
        "and tends NOT to carry. The product pressure-z x (-thickness-z) is "
        "large exactly in the unsupported cell and signed by the pressure "
        "direction, so a positive factor value = upward pressure in a thin "
        "book -> up drift expected. Neither leg alone carries this: bare "
        "slope state and bare microprice state are both dead on 588000."
    ),
    info_set="microprice_dev, book_slope",
    inspiration=(
        "iter-003 R4-D brief direction (d) microprice_dev vs book_slope "
        "disagreement (price pressure without slope support); Zovko & "
        "Farmer (2002) book-profile depth accumulation; round-1 deaths of "
        "book_slope_z_300s / microprice_dev_z_300s motivate the "
        "interaction-only form."
    ),
    compute=compute,
)
