"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_bid_arrival_x_buyvol: hidden bid-support VELOCITY x z(buy_vol_60s) --
hidden demand actively ARRIVING at the same moment buy aggression runs
(stock and flow building together).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing z windows
LAG = 20  # 20 x 3s rows = 60s velocity lag


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _hidden_bid_share() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    return (
        pl.when(tb > 0.0)
        .then(hb / tb)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """diff_60s(z(hidden bid share)) x z(buy_vol_60s); warm-up null."""
    vel = _z(_hidden_bid_share(), W).diff(LAG)
    buy_z = _z(pl.col("buy_vol_60s").cast(pl.Float64), W)
    return part.select((vel * buy_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_bid_arrival_x_buyvol",
    mechanism=(
        "Arrival-confirmed demand: the 60s VELOCITY of the hidden-bid-"
        "support z multiplied by the z of buy volume. This fires when "
        "hidden bid depth is actively BUILDING (fresh patient orders "
        "migrating below the touch) at the same time aggressive buying is "
        "elevated -- stock and flow rising together. A hidden reservoir "
        "that is merely high (the level interaction hidden_bid_x_buyvol_z) "
        "may be stale positioning; one that is ARRIVING while aggression "
        "buys is demand committing NOW on both the displayed (trade) and "
        "undisplayed (queue) channels simultaneously -- the sharpest "
        "confirmation of an ongoing bid regime, expected to continue at "
        "15-60s. Negative values mark conflicting states (arrival with "
        "absent buying, or withdrawal during buying) where the demand "
        "regime is internally inconsistent and prone to stall. Doubly "
        "dynamic by construction: a derivative of the hidden state gated "
        "by a flow regime, distinct from every admitted level/occupancy "
        "member of the R3-B cluster."
    ),
    info_set="total_bid_vol, depth_bid5, buy_vol_60s (batch-2)",
    inspiration=(
        "iter-003 R4-B brief directions (a) x (c): crossing the hidden-"
        "side velocity with side-attributed flow; dynamics-of-the-winning-"
        "cluster mandate of round 4."
    ),
    compute=compute,
)
