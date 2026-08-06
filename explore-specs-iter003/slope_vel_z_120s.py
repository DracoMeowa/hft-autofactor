"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

slope_vel_z_120s: regime-normalized steepening/thinning VELOCITY of the
book-shape profile -- z-scored 60s change of book_slope.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20   # 20 x 3s rows = 60s velocity step
W = 40   # 40 x 3s rows = 120s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(book_slope.diff(20), 120s); warm-up null (diff then z propagate)."""
    vel = pl.col("book_slope").diff(D)
    return part.select(_z(vel, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="slope_vel_z_120s",
    mechanism=(
        "Liquidity-regime SHIFT SPEED, regime-normalized: book_slope "
        "tracks how fast depth accumulates away from the touch; its 60s "
        "change is the steepening/thinning velocity of the whole depth "
        "profile. Raw slope velocity died in round 1 (book_slope_delta_60s) "
        "because the RAW scale mixes the instrument's baseline shape noise "
        "with genuine regime moves. Z-scoring the velocity against its own "
        "trailing-120s distribution keeps only unusually FAST "
        "steepening/thinning events relative to recent shape dynamics -- "
        "the round-2 revival pattern (fast columns work when "
        "regime-normalized: ofi_15s_z_120s passed all 5 horizons while raw "
        "ofi_accel_15_60 hit the panel wall). A burst of rapid thinning "
        "(depth profile flattening fast) is liquidity being withdrawn -- a "
        "regime shift that changes imminent price impact; rapid steepening "
        "is fresh commitment arriving. The direction of the IC resolves "
        "which regime shift dominates at 15-60s, but the information "
        "question (unusual SPEED of shape change) is untouched by the dead "
        "level-z and raw-delta forms."
    ),
    info_set="book_slope",
    inspiration=(
        "iter-003 R4-D brief direction (b) steepening/thinning velocity "
        "(liquidity regime shift); round-2 lesson that regime-normalized "
        "fast dynamics survive where raw fast deltas die."
    ),
    compute=compute,
)
