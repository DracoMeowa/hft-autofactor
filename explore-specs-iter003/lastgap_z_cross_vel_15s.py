"""Explore-lane prototype spec (iter-003 R5, family R5-A).

lastgap_z_cross_vel_15s: z-level vs instantaneous-velocity divergence on
the last-trade-to-mid gap (ticks), CROSSING form -- the 300s z of the gap
crossed zero within the last 15s; value is the z-velocity, only at
crossings, else 0. Aggressor-side regime reversal events.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s crossing lookback

#: SSE ETF minimum price increment (e.g. 588000): 0.001 RMB per tick
TICK = 0.001


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _lastgap() -> pl.Expr:
    """(last_px - mid_px) / TICK in ticks; positive = buyer aggression."""
    return (pl.col("last_px") - pl.col("mid_px")) / TICK


def compute(part: pl.DataFrame) -> pl.Series:
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the aggressor-side regime flip.
    """
    z = _z(_lastgap(), W)
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
    name="lastgap_z_cross_vel_15s",
    mechanism=(
        "Aggressor-side regime reversal events: the trailing-300s z of "
        "the last-trade-to-mid gap (ticks) crosses zero within 15s. The "
        "gap (last minus mid, in ticks) is the fastest directional "
        "microstructure signal on the panel: positive means the last "
        "trade lifted the ask (buyer aggression), negative means it hit "
        "the bid (seller aggression). A zero-crossing of the 300s z "
        "means the aggressor-side bias has just flipped from "
        "predominantly buyer- to predominantly seller-initiated (or vice "
        "versa) relative to the recent 300s norm -- the side hitting "
        "the book has changed hands. The crossing VELOCITY scores how "
        "decisive the hand-off is: a fast, forceful cross marks informed "
        "re-direction of executable flow (a program switching from "
        "buying to selling), whose new direction continues at 15-60s "
        "before the book re-equilibrates. Event-sparse (0 off "
        "crossings). DEDUP: library last_mid_gap_ticks is the RAW "
        "instantaneous gap (single-row, no rolling context); here the "
        "gap is z-scored over 300s and only the zero-crossing EVENT is "
        "scored. The gap z crossing zero is a different object from the "
        "raw gap sign: the raw gap flips sign every few rows (noise), "
        "but the z crossing zero means the SMOOTHED aggressor bias has "
        "changed regime, which is far more informative."
    ),
    info_set="last_px, mid_px",
    inspiration=(
        "iter-003 R5-A family brief: apply the crossing template to the "
        "last-mid gap (constructed inline from last_px, mid_px); the "
        "library factor last_mid_gap_ticks gives the raw gap level but "
        "the z-vs-velocity crossing form is untested."
    ),
    compute=compute,
)
