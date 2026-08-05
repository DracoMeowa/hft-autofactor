"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

top5div_x_extremes: hidden-depth divergence regime gated by day-range
EXTREMENESS -- mismatch structure is only tested at the resolution zones.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100          # 100 x 3s rows = 300s trailing window for the divergence z
EPS = 1e-12


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(wdi - full-book imbalance, 300s) x |day_range_pos - 0.5| x 2."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    div_z = _z(pl.col("wdi") - fbi, W)
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    ext = (pos - 0.5).abs() * 2.0
    return part.select((div_z * ext).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="top5div_x_extremes",
    mechanism=(
        "Resolution-zone gate on the hidden-depth divergence signal. "
        "Mid-range, the mismatch between displayed touch strength and "
        "deep backing is transient structure that never gets tested; AT "
        "the day's extremes it is put on trial. Thin-backed displayed "
        "strength (div_z > 0: touch stronger than the deep book) at a "
        "day extreme is a level that cannot survive probing: rejection at "
        "the high or breakdown through the low -- both DOWN. Deep-backed "
        "structure (div_z < 0: reservoir behind a modest touch) holds: "
        "the high consolidates/extends or the low bounces -- both UP. "
        "Both zones imply NEGATIVE IC of div_z x extremeness "
        "(|pos-0.5| x 2). The AMPLITUDE gate is the key structural "
        "difference from the round-2-dead range_pos_x_ofi_z / "
        "range_pos_x_wdi (linear centered products, IS-dead): mid-range "
        "mismatch rows compress toward zero instead of sign-flipping, so "
        "the factor localizes divergence information at the resolution "
        "zones -- a re-ranking, not a rescaling of the parent "
        "top5_book_div_z_300s."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol, mid_px",
    inspiration=(
        "iter-003 R3-D family brief direction 2 (hidden-depth signal may "
        "matter most near range extremes); round-2 champion "
        "top5_book_div_z_300s x round-1 champion mid_day_range_pos; "
        "round-2 death map: linear range-pos x flow products IS-dead -> "
        "use an extremeness amplitude gate instead."
    ),
    compute=compute,
)
