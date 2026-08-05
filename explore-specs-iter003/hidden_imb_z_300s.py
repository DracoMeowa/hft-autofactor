"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

hidden_imb_z_300s: trailing-300s z-score of the hidden-layer imbalance --
(hidden_bid - hidden_ask) / (hidden_bid + hidden_ask), using ONLY depth
beyond the top-5 levels. Full-book imbalance refined to the hidden layer.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _hidden_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    den = hb + ha
    # Zero hidden depth on both sides -> neutral 0.0 (no hidden skew). Kept a
    # real value (not null) so the trailing-window z sees a complete series.
    return pl.when(den > 0.0).then((hb - ha) / den).otherwise(pl.lit(0.0))


def compute(part: pl.DataFrame) -> pl.Series:
    """z(hidden-layer imbalance, 300s); warm-up rows null."""
    return part.select(_z(_hidden_imb(), W).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_z_300s",
    mechanism=(
        "Hidden-layer pressure regime: the bid/ask imbalance computed ONLY "
        "from depth beyond the executable top-5 levels (total minus top-5 on "
        "each side), z-scored against its trailing-300s distribution. "
        "fullbook_imb_z_300s blends the touch into the total; this strips "
        "the executable layer out and reads the queued-intent layer alone. "
        "Hidden depth is posted deliberately and is not forced to execute, so "
        "a persistently hidden-bid-skewed book is patient accumulation that "
        "the displayed book can mask -- intent-bearing positioning that "
        "precedes visible queue changes and conditions drift at 300-900s. "
        "The ratio form is mandatory: raw hidden-quantity momentum IS-dead "
        "in round 2, but the same layer normalized into a bounded imbalance "
        "and regime-z-scored is a different, live class of statistic."
    ),
    info_set="total_bid_vol, total_ask_vol, depth_bid5, depth_ask5 (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 5 (full-book imbalance refinement "
        "using only hidden depth, in ratio form); round-2 lesson: hidden "
        "depth carries real info only in ratio/z form (hidden_imb momentum "
        "died, fullbook_imb_z lived)."
    ),
    compute=compute,
)
