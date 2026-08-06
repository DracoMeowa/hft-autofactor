"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

slope_z_cross_vel_15s: fresh liquidity-regime SWITCH events on book_slope --
the 300s z of book_slope crossed zero within the last 15s; value is the
z-velocity, only at crossings, else 0. Event-sparse by construction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing z window on book_slope
LAG = 5   # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0."""
    z = _z(pl.col("book_slope"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((flip * (z - z_lag)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="slope_z_cross_vel_15s",
    mechanism=(
        "The moment the liquidity-thickness regime turns: book_slope "
        "measures how fast depth accumulates away from the touch (thick "
        "vs thin book walls). When its trailing-300s z crosses zero within "
        "the last 15s, the book's five-minute thickness regime has just "
        "flipped -- depth that had been unusually concentrated/dispersed "
        "for minutes reversed to the opposite state in seconds. Such fast "
        "shape reversals mark active repositioning by limit-order "
        "providers (quote-layer rebuilds, liquidity commitment switching "
        "sides of its norm) and change the impact environment for the "
        "aggression that follows. The VELOCITY of the crossing (z now "
        "minus z 15s ago) grades how decisive the flip is: a sharp, "
        "forceful regime change re-prices the impact environment more than "
        "a slow drift through zero. The factor is event-sparse -- exactly "
        "0 except right after crossings -- so it is a different object "
        "from the dead book_slope_z_300s (always-on level z) and from the "
        "dead book_slope_delta_60s (always-on raw delta); the template "
        "mirrors ofi_z_cross_vel_15s, which passed 4 horizons in round 3, "
        "applied to the shape column instead of the flow column."
    ),
    info_set="book_slope",
    inspiration=(
        "iter-003 R4-D brief direction (b) book-shape regime shift; the "
        "admitted ofi_z_cross_vel_15s template (R3-C lesson: z-level vs "
        "velocity disagreement is signal) applied to book_slope itself, "
        "which R4-C explicitly leaves to R4-D."
    ),
    compute=compute,
)
