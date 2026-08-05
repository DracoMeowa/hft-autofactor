"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ofi_z_cross_vel_15s: fresh book-flow REGIME-SWITCH events -- the 300s z of
ofi_60s crossed zero within the last 15s; value is the z-velocity, only at
crossings, else 0. Event-sparse by construction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window on ofi_60s
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    Warm-up rows null (z warm-up propagates through the shift and the
    flip indicator); non-crossing rows are exactly 0, crossing rows carry
    the signed velocity of the regime change.
    """
    z = _z(pl.col("ofi_60s"), W)
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
    name="ofi_z_cross_vel_15s",
    mechanism=(
        "The moment a book-flow regime turns: when the trailing-300s z of "
        "ofi_60s crosses zero within the last 15s, the book's five-"
        "minute flow regime has just changed sign -- an unusually "
        "information-dense event, because a regime that was one-sided "
        "for minutes reversed in seconds. The VELOCITY of the crossing "
        "(z now minus z 15s ago, signed by the new direction) measures "
        "how decisive the flip is: a fast, forceful cross marks a sharp "
        "reallocation of limit interest (informed direction change, "
        "queue pull-and-rebuild) whose new direction continues at "
        "15-60s; a slow drift across zero scores weak. The factor is "
        "event-sparse -- exactly 0 except in the rows immediately after "
        "a crossing -- so it is a different object from the dead "
        "ofi_mom_60s, which is z-momentum ACTIVE EVERY ROW (always-on "
        "momentum of z-ofi at a 60s lag): here momentum is measured "
        "only at sign-crossing events over a 15s span. The crossing "
        "test is causal by construction: current z vs lagged z, no "
        "lookahead."
    ),
    info_set="ofi_60s (library)",
    inspiration=(
        "iter-003 R3-C brief direction 5 (OFI regime switches: z-score "
        "crossing events approximated causally via rolling comparison of "
        "current z vs lagged z); the dead ofi_mom_60s motivates the "
        "event-conditional (not always-on) construction."
    ),
    compute=compute,
)
