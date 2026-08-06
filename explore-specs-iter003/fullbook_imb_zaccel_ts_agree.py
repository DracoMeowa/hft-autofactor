"""Explore-lane prototype spec (iter-003 R6D, family R6D).

fullbook_imb_zaccel_ts_agree: timescale CONFIRMATION gate on the
full-book imbalance z-acceleration. Fast 15s z-acceleration amplified by
slow 60s z-acceleration magnitude, ONLY when both accelerations agree;
zero on disagreement. Weighted by |z|.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG15 = 5
LAG60 = 20


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (sum); null when denominator is 0."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z15 * |d2z60| * |z| when signs agree; 0 on disagreement.

    Warm-up rows null (nulls propagate through the sign comparison).
    """
    z = _z(_fullbook_imb(), W)
    dz15 = z - z.shift(LAG15)
    dz60 = z - z.shift(LAG60)
    d2z15 = dz15 - dz15.shift(LAG15)
    d2z60 = dz60 - dz60.shift(LAG60)
    agree = (d2z15.sign() == d2z60.sign()).cast(pl.Float64)
    return part.select((d2z15 * d2z60.abs() * z.abs() * agree).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zaccel_ts_agree",
    mechanism=(
        "Timescale-confirmation acceleration on the broad book: the fast "
        "15s z-acceleration of z_300(full-book imbalance), amplified by "
        "the slow 60s z-acceleration magnitude, emitted ONLY when both "
        "accelerations agree in direction. The full-book imbalance sees "
        "passive depth beyond level 5 -- outer-queue institutional tilt. "
        "When BOTH 15s and 60s curvature confirm the same direction while "
        "the broad regime is already stretched (high |z|), the consensus "
        "marks a mature multi-horizon acceleration in the outer queue: "
        "large passive positions being repositioned at increasing speed "
        "over both seconds and a full minute. The product form scales "
        "with confirmation strength on both scales. Disagreement rows are "
        "exactly zero: cross-timescale conflict in the outer queue is "
        "treated as noise. Distinct from fullbook_imb_zaccel_15s_x_60s "
        "(which fires even on disagreement): this is a stricter "
        "confirmation-only gate, sparser, carrying signal only when the "
        "full-book acceleration is unambiguously multi-horizon."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6D family brief direction 1: agreement-gated "
        "cross-timescale acceleration on the full-book base."
    ),
    compute=compute,
)
