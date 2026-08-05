"""Explore-lane prototype spec (iter-003, depth-book lens).

queue_pressure_x_slope: interaction -- z-scored 60s top-of-book
imbalance delta x z-scored book_slope. Directional touch rebuild that
agrees with the deep book's shape gradient.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20    # 20 x 3s rows = 60s top-book delta
W = 100   # 100 x 3s rows = 300s z window


def _cz(x, w: int):
    """Canonical causal z-score: constant window -> 0.0, warm-up null."""
    mean = x.rolling_mean(window_size=w, min_samples=w)
    std = x.rolling_std(window_size=w, min_samples=w)
    z = (x - mean) / std
    return (
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
    )


def compute(part: pl.DataFrame) -> pl.Series:
    bq = pl.col("bid1_qty").cast(pl.Float64)
    aq = pl.col("ask1_qty").cast(pl.Float64)
    tot = bq + aq
    tb = (
        pl.when(tot > 0.0)
        .then((bq - aq) / tot)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    z_touch = _cz(tb.diff(D), W)      # regime-adjusted touch-queue rebuild
    z_shape = _cz(pl.col("book_slope"), W)  # regime-adjusted book shape
    return part.select((z_touch * z_shape).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="queue_pressure_x_slope",
    mechanism=(
        "Supported vs unsupported rebuild (interaction): the product of the "
        "regime-adjusted 60s top-of-book imbalance delta and the "
        "regime-adjusted book shape. A touch-queue rebuild in the SAME "
        "signed direction as the book's shape gradient is backed by the "
        "committed deep structure - the pressure should propagate into "
        "continuation; a rebuild AGAINST the shape gradient is shallow and "
        "fragile - likely to fade. Neither component alone captures the "
        "conditional agreement; interactions are the dimension the archive "
        "flags as underexploited after levels and simple deltas."
    ),
    info_set="bid1_qty, ask1_qty, book_slope (library factor)",
    inspiration=(
        "iter-003 family brief: interaction of top-book dynamics with book "
        "shape; prem_x_ofi precedent (interactions are a real, orthogonal "
        "information dimension on this panel); queue-reactive impact "
        "(Cont-Stoikov-Talreja 2010) conditioned on the depth profile."
    ),
    compute=compute,
)
