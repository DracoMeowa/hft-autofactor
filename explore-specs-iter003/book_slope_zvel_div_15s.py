"""Explore-lane prototype spec (iter-003 R4, family R4-C).

book_slope_zvel_div_15s: z-level vs instantaneous-velocity divergence on
book_slope, SIGNED-DIFFERENCE form -- the slow shape-regime z minus its own
fast z-velocity, itself regime-normalized.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(book_slope, 300s) - z(dz, 300s) where dz = 15s z-velocity.

    Warm-up rows null: the z warm-up propagates through dz into the
    velocity's own trailing z.
    """
    z_e = _z(pl.col("book_slope"), W)
    dz_e = z_e - z_e.shift(LAG)
    tmp = part.select(z_e.alias("_z"), dz_e.alias("_dz"))
    tmp = tmp.select(pl.col("_z"), _z(pl.col("_dz"), W).alias("_dzz"))
    return tmp.select((pl.col("_z") - pl.col("_dzz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="book_slope_zvel_div_15s",
    mechanism=(
        "Book-shape regime overextension vs its own fast edge: z_300("
        "book_slope) minus the trailing-300s z of its own 15s z-velocity. "
        "book_slope (mean ln-cum-depth vs distance slope across both "
        "sides) is a shape, not a signed imbalance, so the information "
        "must live in its DYNAMICS: when the shape regime z is high but "
        "the fast edge is already moving down (positive divergence), the "
        "deep-book liquidity structure that had been building is being "
        "pulled RIGHT NOW -- hollowing of the depth curve precedes the "
        "price move it was cushioning, and price drifts against the "
        "level direction at 15-60s; when normalized velocity leads level, "
        "the structure is still accumulating and supports continuation. "
        "DEDUP: a different object from the dead book_slope_z_300s (bare "
        "level z) and book_slope_delta_60s (raw delta) -- the velocity "
        "enters regime-normalized and subtracted from the level, a "
        "relative-stretch question neither parent asks, and no library "
        "factor touches book_slope at all."
    ),
    info_set="book_slope",
    inspiration=(
        "iter-003 R4-C family brief: signed-divergence form of the "
        "admitted ofi_z_cross_vel_15s z-vs-velocity template; for a "
        "non-directional shape column the level-vs-velocity tension is "
        "the natural way to recover directional content from dynamics."
    ),
    compute=compute,
)
