"""Explore-lane prototype spec (iter-003 R5, family R5-A).

ltns_z_cross_vel_15s: z-level vs instantaneous-velocity divergence on
large_trade_net_share_60s (signed net share of the largest ~10% of
prints), CROSSING form -- the 300s z of ltns crossed zero within the
last 15s; value is the z-velocity, only at crossings, else 0.
Institutional large-trade direction regime reversal events.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the large-trade direction regime flip.
    """
    z = _z(pl.col("large_trade_net_share_60s"), W)
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
    name="ltns_z_cross_vel_15s",
    mechanism=(
        "Large-trade direction regime reversal events: the trailing-300s "
        "z of large_trade_net_share_60s crosses zero within 15s. ltns "
        "attributes a SIGNED net direction to the largest ~10% of prints "
        "(+ when biggest tickets are buyer-initiated, - when seller-"
        "initiated). Unlike the direction-free large_trade_share_60s "
        "whose LEVEL died in iter-002 (no sign = no directional "
        "information), ltns carries the direction of whale flow. A "
        "zero-crossing of its 300s z means the institutional big-ticket "
        "flow has just switched from net-buying-above-norm to net-"
        "selling-above-norm (or vice versa) -- the whales have changed "
        "direction. The crossing VELOCITY scores how decisive the "
        "hand-off is: a fast cross marks a program switching sides "
        "(information-driven re-direction), whose new direction "
        "continues at 15-60s while the program executes. Event-sparse "
        "(0 off crossings). DEDUP: library ltns_z_180s is the pure "
        "LEVEL z (180s window, state only, no velocity); library "
        "ltns_delta_60s is a raw unnormalized 60s delta; library "
        "ltns_confirms_ofi_z and ltns_confirms_ti15_z cross ltns with "
        "OTHER flow signals. Here the ltns z is crossed with ITS OWN "
        "velocity, and only the regime-reversal EVENT is scored -- a "
        "different economic question from all four library factors."
    ),
    info_set="large_trade_net_share_60s (batch-2)",
    inspiration=(
        "iter-003 R5-A family brief: apply the crossing template to "
        "large_trade_net_share_60s (the SIGNED large-trade column that "
        "carries institutional direction); library ltns_z_180s gives "
        "the level state but the z-vs-velocity crossing form is "
        "untested."
    ),
    compute=compute,
)
