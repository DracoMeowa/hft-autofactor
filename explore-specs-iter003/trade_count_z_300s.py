"""Explore-lane prototype spec (iter-003 R2-C, trade-structure lens).

trade_count_z_300s: trailing-300s z-score of the number of trades per
60s -- the participation/attention regime (many small prints vs few).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(n_trades_60s, 300s); warm-up rows null."""
    return part.select(_z(pl.col("n_trades_60s"), W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="trade_count_z_300s",
    mechanism=(
        "Participation regime: the number of trades per minute, z-scored "
        "against its trailing-300s distribution, measures arrival "
        "intensity -- how many separate decisions are hitting the tape. A "
        "burst of MANY small trades is the signature of retail/algorithmic "
        "attention, while the marginal price-setters on this ETF are "
        "institutional; noise-flow bursts get absorbed and traded against "
        "by market makers (noise-trader risk literature), so unusually "
        "high count z is hypothesized CONTRARIAN: reversion at 60-900s "
        "after attention bursts, while count droughts mark thin "
        "participation where any institutional flow dominates. Count is "
        "the arrival-intensity axis, complementary to the ticket-scale "
        "axis of the size family -- together they span the trade-"
        "structure state space."
    ),
    info_set="n_trades_60s (wishlist batch 1)",
    inspiration=(
        "iter-003 R2-C family brief direction 5 (trade granularity: "
        "n_trades z); retail-attention contrarian hypothesis (De Long, "
        "Shleifer, Summers & Waldmann 1990 noise traders); arrival-"
        "intensity axis distinct from ticket-size axis."
    ),
    compute=compute,
)
