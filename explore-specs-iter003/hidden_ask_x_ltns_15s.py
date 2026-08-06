"""Explore-lane prototype spec (iter-003 R5-D, hidden-depth x trade flow).

hidden_ask_x_ltns_15s: hidden ask-supply z x large-trade net share --
patient queued supply backing the INFORMED large-trade direction. Fires
when institutional-size prints lean sell AND a deep hidden ask overhang
sits above the touch.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window


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
    """z(hidden_ask_share, 300s) x large_trade_net_share_60s; warm-up null."""
    hid_z = _z(_hidden_ask_share(), W)
    ltns = pl.col("large_trade_net_share_60s")
    return part.select((hid_z * ltns).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_ask_x_ltns_15s",
    mechanism=(
        "Institutional supply backing: the 300s z of the hidden-ask-"
        "supply share multiplied by the signed net share of the "
        "largest ~10% trades. Fires when large traders net sell WHILE "
        "an unusually deep patient-ask overhang sits hidden above the "
        "touch -- supply committing on the INFORMED channel (large "
        "orders) AND on the PATIENT channel (queued overhang that "
        "will cap rallies) simultaneously. The mechanism: "
        "institutional-size prints reveal where size is being "
        "distributed; a hidden ask that is unusually deep at the same "
        "time reveals a queued ceiling ready to refill -- the "
        "combination is a doubly-confirmed distribution regime that "
        "caps rallies and pressures the downside at 15-60s. The "
        "product is signed by the large-trade direction: when net "
        "large selling coincides with a deep hidden ask, the product "
        "is negative (high ask_z x negative ltns), marking confirmed "
        "supply pressure. Distinct from hidden_bid_x_ltns_15s (the "
        "bid-side twin): the two use independent hidden parents and "
        "fire during different regime combinations -- both can be "
        "nonzero during heavy two-sided institutional trading. Distinct "
        "from the round-4 hidden_ask_x_sellvol_z (hidden ask x ALL "
        "sell volume): large-trade net share isolates the informed-"
        "size component, distinct from the broad sell-volume level."
    ),
    info_set=(
        "total_ask_vol, depth_ask5, large_trade_net_share_60s (batch-2)"
    ),
    inspiration=(
        "iter-003 R5-D brief direction (b): hidden supply x "
        "large_trade_net_share_60s; informed-size proxy coupling, "
        "supply-side twin of the bid large-trade interaction."
    ),
    compute=compute,
)
