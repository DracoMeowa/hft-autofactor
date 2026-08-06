"""Explore-lane prototype spec (iter-003 R5-D, hidden-depth x trade flow).

hidden_bid_x_ltns_15s: hidden bid-support z x large-trade net share --
patient queued demand backing the INFORMED large-trade direction. Fires
when institutional-size prints lean buy AND a deep hidden bid reservoir
sits below the touch.
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
    """z(hidden_bid_share, 300s) x large_trade_net_share_60s; warm-up null."""
    hid_z = _z(_hidden_bid_share(), W)
    ltns = pl.col("large_trade_net_share_60s")
    return part.select((hid_z * ltns).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_bid_x_ltns_15s",
    mechanism=(
        "Institutional demand backing: the 300s z of the hidden-bid-"
        "support share multiplied by the signed net share of the "
        "largest ~10% trades. Fires when large traders net buy WHILE an "
        "unusually deep patient-bid reservoir sits hidden below the "
        "touch -- demand committing on the INFORMED channel (large "
        "orders are the canonical institutional proxy) AND on the "
        "PATIENT channel (queued depth that will absorb and refill) "
        "simultaneously. The mechanism: institutional-size prints "
        "reveal where size is being deployed; a hidden bid that is "
        "unusually deep at the same time reveals an iceberg ready to "
        "replenish -- the combination is a doubly-confirmed demand "
        "regime (aggressive + patient backing) that persists at "
        "15-60s. Distinct from the round-4 hidden_bid_x_buyvol_z "
        "(hidden bid x ALL buy volume): large-trade net share isolates "
        "the informed-size component, ignoring retail-size noise that "
        "dominates total volume. The large-trade net share is signed "
        "(positive = net large buying, negative = net large selling), "
        "so the product is signed by the large-trade direction; the z "
        "on hidden depth normalizes the regime. Not a sign-flip of "
        "hidden_ask_x_ltns_15s: the two use independent hidden-side "
        "parents (bid vs ask share) and can both fire during heavy "
        "two-sided institutional trading."
    ),
    info_set=(
        "total_bid_vol, depth_bid5, large_trade_net_share_60s (batch-2)"
    ),
    inspiration=(
        "iter-003 R5-D brief direction (b): hidden support x "
        "large_trade_net_share_60s; institutional-size prints as the "
        "informed-flow proxy, distinct from the all-volume interactions "
        "of round 4."
    ),
    compute=compute,
)
