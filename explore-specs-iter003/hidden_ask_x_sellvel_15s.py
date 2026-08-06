"""Explore-lane prototype spec (iter-003 R5-D, hidden-depth x trade flow).

hidden_ask_x_sellvel_15s: hidden ask-supply LEVEL x 15s VELOCITY of
z-scored sell volume -- patient ask overhang sitting while sell aggression
actively ARRIVES (supply-regime stock + flow-rising confirmation).
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


def _hidden_ask_share() -> pl.Expr:
    """Hidden ask volume as a share of total ask volume."""
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    return (
        pl.when(ta > 0.0)
        .then(ha / ta)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(hidden_ask_share, 300s) x diff(z(sell_vol_60s, 300s), 5)."""
    hid_z = _z(_hidden_ask_share(), W)
    sell_z = _z(pl.col("sell_vol_60s").cast(pl.Float64), W)
    sell_vel = sell_z.diff(LAG)
    return part.select((hid_z * sell_vel).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_ask_x_sellvel_15s",
    mechanism=(
        "Arrival-confirmed supply via the TRADE channel: the 300s z of "
        "the hidden-ask-supply share multiplied by the 15s VELOCITY of "
        "the z-scored sell volume. Fires when an unusually deep patient-"
        "ask overhang sits above the touch WHILE sell aggression is "
        "actively rising -- the sell-side velocity is the fresh input. "
        "The mechanism: a hidden ask that is merely high may be a stale "
        "ceiling; pairing it with RISING sell volume detects "
        "distribution committing NOW on both the visible (aggressor) "
        "and undisplayed (queued) channels -- queued supply refilling "
        "the offer while sellers step in aggressively, a doubly-"
        "confirmed distribution regime that caps rallies and "
        "pressures the downside at 15-60s. Supply-side twin of "
        "hidden_bid_x_buyvel_15s (not a sign-flip: the two sides "
        "rotate independently and can both be confirmed during heavy "
        "two-sided trading). Side-attributed GROSS sell volume carries "
        "information the dead NET-TI interactions (round 3) are "
        "structurally blind to. Distinct from hidden_ask_x_sellvol_z "
        "(round-4 admitted level x level): velocity on the trade side "
        "detects arrival episodes the level interaction misses."
    ),
    info_set="total_ask_vol, depth_ask5, sell_vol_60s (batch-2)",
    inspiration=(
        "iter-003 R5-D brief direction (b): hidden_ask_supply x "
        "sell_vol_60s velocity/sign; supply-side twin of the bid "
        "velocity interaction."
    ),
    compute=compute,
)
