"""Explore-lane prototype spec (iter-003 R6, family R6B).

fullbook_imb_zjerk_extreme_15s: JERK-extremity product on the full-book
imbalance. The 3rd difference of z (15s jerk = diff of acceleration),
weighted by level extremity |z|. Abrupt curvature changes in the broadest
deep-book positioning regime.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity, acceleration, jerk


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (sum); null if no book volume."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """d3z * |z| where d3z = 15s jerk of z (3rd derivative).

    Warm-up rows null.
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    d3z = d2z - d2z.shift(LAG)
    return part.select((d3z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zjerk_extreme_15s",
    mechanism=(
        "Jerk-weighted broad-book stretch: the 15s jerk of z_300(full-book "
        "imbalance) -- the 3rd difference, the change rate of the "
        "acceleration -- weighted by how extreme the regime being jerked is "
        "(|z|). The full-book imbalance reaches beyond level 5 and reflects "
        "slow-building patient positioning; its acceleration measures "
        "whether the broad build is speeding up, and its JERK measures "
        "whether that acceleration just abruptly reversed -- the moment a "
        "committed multi-level institutional build snaps from intensifying "
        "to unwinding (or the reverse). When such a curvature break lands "
        "on a stretched deep-book regime (high |z|), it flags a regime "
        "change in the broadest positioning layer and 15-60s returns follow "
        "the new direction. Distinct from library fullbook_imb_zvel_div_15s "
        "(level minus velocity) and fullbook_imb_zvel_extreme_15s (velocity "
        "x |level|): jerk is the 3rd derivative, capturing curvature BREAKS "
        "that the lower derivatives smooth over -- a steadily-accelerating "
        "extreme build reads ~0 here."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6-B family brief: JERK (3rd derivative) extremeness on "
        "the full-book imbalance; the slow-build deep positioning makes an "
        "abrupt curvature reversal a strong regime-change marker."
    ),
    compute=compute,
)
