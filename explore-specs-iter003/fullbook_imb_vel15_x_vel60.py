"""Explore-lane prototype spec (iter-003 R5, family R5-B).

fullbook_imb_vel15_x_vel60: NEW construction -- cross-timescale velocity
mismatch on the full-book imbalance. 15s z-velocity signed by 60s
z-velocity direction. Tests whether broad-book flow alignment across
two timescales predicts continuation on the full visible book.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 300s trailing z window
LAG15 = 5   # 15s fast velocity
LAG60 = 20  # 60s slow velocity


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """dz15 * sign(dz60) on the full-book imbalance z-regime.

    Warm-up rows null (z warm-up propagates through the 60s shift).
    """
    z = _z(_fullbook_imb(), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    return part.select((dz15 * dz60.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_vel15_x_vel60",
    mechanism=(
        "Cross-timescale velocity confirmation on the broad book: the "
        "15s z-velocity of z_300(fullbook_imb) multiplied by sign of the "
        "60s z-velocity. The full book (batch-2 total_*_vol, wider than "
        "depth_*5) aggregates patient deeper-queue positioning that "
        "normally moves slowly; when the 15s and 60s velocities of its "
        "imbalance regime AGREE, broad liquidity is relocating in a "
        "sustained, multi-horizon fashion -- institutional limit interest "
        "moving the whole book one way, and the fast velocity magnitude "
        "carries continuation at 15-60s. Disagreement (fast vs slow "
        "opposing) flips the sign, encoding that the slow broad-book "
        "trend dominates. Distinct from library fullbook_imb_z_cross_vel_15s "
        "(round-4 single-horizon sign-flip event): the economic input "
        "here is cross-horizon velocity AGREEMENT, not the zero-crossing "
        "event."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R5-B family brief: cross-timescale velocity mismatch "
        "on the full-book base; broad-book positioning is inherently "
        "slower, so the 15s/60s agreement isolates truly sustained "
        "repositioning."
    ),
    compute=compute,
)
