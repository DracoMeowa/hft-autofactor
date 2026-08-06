"""Explore-lane prototype spec (iter-003 R5-D, hidden-depth x trade flow).

hidden_bid_x_buyvel_15s: hidden bid-support LEVEL x 15s VELOCITY of
z-scored buy volume -- patient bid depth sitting while buy aggression
actively ARRIVES (stock + flow-rising confirmation).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z windows
LAG = 5  # 5 x 3s rows = 15s velocity lag


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _hidden_bid_share() -> pl.Expr:
    """Hidden bid volume as a share of total bid volume."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    return (
        pl.when(tb > 0.0)
        .then(hb / tb)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(hidden_bid_share, 300s) x diff(z(buy_vol_60s, 300s), 5); warm-up null."""
    hid_z = _z(_hidden_bid_share(), W)
    buy_z = _z(pl.col("buy_vol_60s").cast(pl.Float64), W)
    buy_vel = buy_z.diff(LAG)
    return part.select((hid_z * buy_vel).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_bid_x_buyvel_15s",
    mechanism=(
        "Arrival-confirmed demand via the TRADE channel: the 300s z of "
        "the hidden-bid-support share multiplied by the 15s VELOCITY of "
        "the z-scored buy volume. Fires when an unusually deep patient-"
        "bid reservoir sits below the touch WHILE buy aggression is "
        "actively rising (not merely elevated) -- the trade-side "
        "velocity is the fresh input. The mechanism: a hidden bid that "
        "is merely high may be stale positioning (the level interaction "
        "hidden_bid_x_buyvol_z from round 4 captures that); pairing it "
        "with RISING buy volume detects demand committing NOW on both "
        "the visible (aggressor) and undisplayed (queued) channels "
        "simultaneously. Distinct from hidden_bid_arrival_x_buyvol "
        "(round-4 admitted), which puts the velocity on the HIDDEN side "
        "and the level on the TRADE side: here the level is on the "
        "hidden side and the velocity on the trade side -- the mirror "
        "operator, capturing the arrival episode from the other channel. "
        "The two fire at different moments: arrival_x_buyvol peaks when "
        "depth migrates in; buyvel peaks when the aggressor rate "
        "accelerates. Side-attributed GROSS buy volume (not net TI) "
        "carries information even in heavy two-sided states where net "
        "trade imbalance nets to zero (round-3 NET-TI hidden "
        "interactions died; round-4 gross-volume interactions lived)."
    ),
    info_set="total_bid_vol, depth_bid5, buy_vol_60s (batch-2)",
    inspiration=(
        "iter-003 R5-D brief direction (b): hidden_bid_support x "
        "buy_vol_60s velocity/sign; velocity-on-trade-side mirror of "
        "round-4's velocity-on-hidden-side construction."
    ),
    compute=compute,
)
