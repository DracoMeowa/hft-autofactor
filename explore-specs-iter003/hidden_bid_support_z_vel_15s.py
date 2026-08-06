"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_bid_support_z_vel_15s: FAST 15s velocity of the admitted hidden-bid-
support regime z -- abrupt stack/pull events of patient bid depth.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window (matches admitted parent)
LAG = 5  # 5 x 3s rows = 15s velocity lag


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
    """diff_15s of z(bid-side hidden share, 300s); warm-up rows null."""
    z = _z(_hidden_bid_share(), W)
    return part.select(z.diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_bid_support_z_vel_15s",
    mechanism=(
        "Abrupt hidden bid stack/pull events: the 15s change of the "
        "trailing-300s z of the bid-side hidden share. Hidden depth does "
        "not usually migrate in small increments -- large patient orders are "
        "posted or yanked discretely, so the FASTEST detectable arrival of "
        "hidden bid support marks a decisive, likely informed commitment of "
        "demand below the touch (pre-positioning ahead of expected upward "
        "pressure), and the fastest withdrawal marks sudden loss of "
        "conviction. Because the event is fresh, its information decays "
        "quickly and should pay off at the shortest horizons (15-30s), "
        "complementing the 60s velocity which integrates over slower "
        "positioning. Same z-state base as the admitted "
        "hidden_bid_support_z_300s but a different clock: instantaneous "
        "shock to the regime rather than the regime itself or its averaged "
        "drift; deltas of slow state live where levels saturate."
    ),
    info_set="total_bid_vol, depth_bid5 (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (a): 15s velocity of the admitted "
        "hidden-side z's; round-1 lesson that 15-30s horizons reward fast "
        "book state; differentiates the admitted R3-B level regime."
    ),
    compute=compute,
)
