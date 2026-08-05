"""Explore-lane prototype spec (iter-003, etf-regime lens).

vol_confirmed_mom: z(per-snapshot volume increment, 100 rows) x sign(20-row
mid momentum).  Volume-confirmed momentum on the snapshot grid, built from
the wishlist column cum_trade_vol (materialized 2026-08-05).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 300s (100 x 3s rows) z window for volume increments
Z_WINDOW = 100
#: 20 x 3s rows = 60s momentum horizon for the direction sign
MOM_ROWS = 20


def compute(part: pl.DataFrame) -> pl.Series:
    """z(volume increment) x sign(60s mid momentum); warm-up rows null."""
    vinc = pl.col("cum_trade_vol").diff()
    mean = vinc.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = vinc.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    zv = (vinc - mean) / std
    zv = pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(zv)
    mom = pl.col("mid_px").log().diff(MOM_ROWS)
    val = zv * mom.sign()
    return part.select(
        pl.when(zv.is_not_null() & mom.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="vol_confirmed_mom",
    mechanism=(
        "Volume-confirmed momentum: short price moves that arrive on "
        "unusually heavy traded volume versus the trailing 300s norm are "
        "information-driven -- informed traders consume liquidity and "
        "volume clusters around information events -- so the just-revealed "
        "direction continues over 15-60s; moves on thin volume are quote "
        "drift and inventory noise, more likely to revert. The factor "
        "multiplies the volume-surprise z by the SIGN of 60s mid momentum, "
        "so heavy-up-momentum scores positive and heavy-down-momentum "
        "negative (positive IC expected). cum_trade_vol gives the exact "
        "per-snapshot traded volume increment (monotone cumulative series), "
        "cleaner than any trade-count proxy; no registered prototype reads "
        "it, so this opens the intraday volume-pacing dimension."
    ),
    info_set="cum_trade_vol, mid_px",
    inspiration=(
        "iter-003 etf-regime brief: cum_trade_vol intraday volume pacing is "
        "untouched; the high-volume return premium and volume-return "
        "interactions (Gervais, Kaniel & Mingelgrin 2001); meta-lesson "
        "from iter-001/002: deltas/interactions of fast state beat levels. "
        "cum_trade_vol materialized 2026-08-05 (wishlist)."
    ),
    compute=compute,
)
