"""Explore-lane prototype spec (iter-003 R4-B, hidden-depth dynamics lens).

hidden_bid_support_z_vel_60s: 60s VELOCITY of the admitted hidden-bid-support
regime z -- hidden bid liquidity ARRIVING vs WITHDRAWING (derivative of the
R3-B winning cluster, not a window-swap of it).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing z window (matches admitted parent)
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
    """diff_60s of z(bid-side hidden share, 300s); warm-up rows null."""
    z = _z(_hidden_bid_share(), W)
    return part.select(z.diff(LAG).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_bid_support_z_vel_60s",
    mechanism=(
        "Arriving vs withdrawing hidden bid support: the 60s change of the "
        "trailing-300s z of the bid-side hidden share (the admitted "
        "hidden_bid_support_z_300s regime, differentiated). The LEVEL says "
        "how deep patient bids are stacked right now; the VELOCITY says "
        "whether that reservoir is actively filling or draining. Hidden bid "
        "support ARRIVING (rising z) is fresh patient demand migrating into "
        "the deep book -- accumulation intent that is not yet visible at the "
        "executable levels -- which precedes continued upward drift as the "
        "reservoir feeds the touch. Support WITHDRAWING (falling z) is the "
        "queue beneath the touch hollowing out before visible bids thin -- "
        "an early-softness signal the level alone cannot give: a high but "
        "draining reservoir and a low but filling one are opposite regimes "
        "at identical z-extremes. This is the derivative of a z-state in "
        "the live ratio/z class, NOT the dead raw-hidden-qty momentum class "
        "(hidden_imb_mom was momentum of absolute raw quantities); deltas of "
        "slow state carry what levels miss per the round-1/2 meta-lesson."
    ),
    info_set="total_bid_vol, depth_bid5 (batch-2)",
    inspiration=(
        "iter-003 R4-B brief direction (a): velocity of the admitted R3-B "
        "hidden-side z's; hidden_bid_support_z_300s was a round-3 R3-B "
        "admission -- this differentiates it (dynamics of the winning "
        "cluster, new economic input = the derivative)."
    ),
    compute=compute,
)
