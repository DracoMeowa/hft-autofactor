"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

range_pos_x_ofi_z: champion interaction -- centered day-range position times
z-scored order-flow imbalance. The SAME flow means different things at
different heights of the day's battle range.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-12
W_OFI = 60  # 60 x 3s rows = 180s z window on order-book-delta flow


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(day_range_pos - 0.5) * z(ofi_60s, 180s); warm-up rows null."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    cpos = pos - 0.5
    ofi_z = _z(pl.col("ofi_60s"), W_OFI)
    return part.select((cpos * ofi_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_pos_x_ofi_z",
    mechanism=(
        "Order-flow imbalance conditioned on where price sits in the day's "
        "range. Buy-heavy flow (positive OFI z) near the day HIGH presses "
        "into the breakout zone: it either ignites the resting buy-stop "
        "cluster above the high (continuation) or gets absorbed by supply "
        "defending the level (rejection); buy-heavy flow near the day LOW "
        "is defense/accumulation at support. Centered position (pos - 1/2) "
        "times OFI z lets one signed factor encode all four zone-flow "
        "combinations. Round 1 proved both halves separately: range "
        "position passed all five horizons nearly orthogonal to everything, "
        "and state-conditioned flow beat bare flow (spread-z x OFI passed "
        "15s while bare spread level died) -- this spec composes the two "
        "winning principles."
    ),
    info_set="mid_px, ofi_60s",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 6 (champion x flow "
        "interactions); composition of round-1 champion mid_day_range_pos "
        "with the ofi_z_x_spread_z conditioning lesson (Cont, Kukanov & "
        "Stoikov 2014 for OFI's information content)."
    ),
    compute=compute,
)
