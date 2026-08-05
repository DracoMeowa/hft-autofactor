"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

book_div_z_120s: trailing-120s z-score of the top-5 vs full-book divergence
(wdi - full-book imbalance) -- a FASTER mismatch clock than the library's
300s z, catching transient touch-vs-queue break-aways as they form.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
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
    """z(wdi - full-book imbalance, 120s); warm-up rows null."""
    div = pl.col("wdi") - _fullbook_imb()
    return part.select(_z(div, W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="book_div_z_120s",
    mechanism=(
        "Fast structural-mismatch clock: the same touch-vs-full-book "
        "divergence as the library's top5_book_div_z_300s but normalized "
        "against a 120s trailing distribution instead of 300s. The shorter "
        "reference window makes the z react to divergence break-aways within "
        "~2 minutes, flagging the ONSET of a touch pulling away from (or "
        "snapping back to) the deep queue while the 300s z is still anchored "
        "to the older regime. Divergence formation is front-loaded: hidden "
        "depth either arrives to back a touch move or fails to, and that "
        "resolution happens on the scale of tens of seconds to a couple "
        "minutes, so a faster z isolates the transition itself rather than "
        "the settled regime. Distinct question (timing of mismatch change) "
        "at a different reference scale, aimed at the 15-60s horizons where "
        "fast book state carried round-1."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R3-B brief direction 1 (divergence z over other windows); "
        "fast book-state deltas carried 15-60s in round 1; complement to "
        "top5_book_div_z_300s at a faster reference window."
    ),
    compute=compute,
)
