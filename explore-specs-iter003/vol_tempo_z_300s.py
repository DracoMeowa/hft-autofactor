"""Explore-lane prototype spec (iter-003 R2-C, trade-structure lens).

vol_tempo_z_300s: z-score of the 30s growth rate of cumulative traded
volume (log difference) -- tape acceleration/deceleration state.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

PACE = 10  # 10 x 3s rows = 30s pace window
W = 100    # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(diff(log cum_trade_vol, 30s), 300s); warm-up rows null."""
    cv = pl.col("cum_trade_vol")
    lc = (
        pl.when(cv > 0.0)
        .then(cv.log())
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    pace = lc.diff(PACE)
    return part.select(_z(pace, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="vol_tempo_z_300s",
    mechanism=(
        "Volume-tempo surprise: the 30s growth rate of cumulative traded "
        "volume (a log difference over 10 rows, so it measures the volume "
        "RATIO between now and 30s ago) z-scored against its trailing-"
        "300s distribution. Tape ACCELERATING beyond its recent regime "
        "flags a crowded participation episode where information-driven "
        "and inventory-driven activity cluster; on this ETF such "
        "overheated bursts exhaust quickly as market-maker liquidity "
        "restocks, so high tempo z is hypothesized to precede short-"
        "horizon REVERSION (negative IC, 30-300s); decelerating tempo "
        "marks participation withdrawal and stall. The 30s pace window "
        "smooths single-snapshot noise and is complementary to "
        "vol_rate_x_ti / vol_confirmed_mom, which sign the per-snapshot "
        "volume surprise by direction instead of measuring the tempo "
        "STATE itself."
    ),
    info_set="cum_trade_vol (wishlist batch 1)",
    inspiration=(
        "iter-003 R2-C family brief direction 7 (volume tempo: short-"
        "window diff of log cum_trade_vol, then z); volume-pace state; "
        "Gervais, Kaniel & Mingelgrin (2001) volume-return interaction; "
        "cum_trade_vol monotone cumulative series gives exact pace."
    ),
    compute=compute,
)
