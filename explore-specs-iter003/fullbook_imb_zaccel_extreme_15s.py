"""Explore-lane prototype spec (iter-003 R6A, family R6A).

fullbook_imb_zaccel_extreme_15s: z-ACCELERATION-extremeness product on the
full-book imbalance. The 15s acceleration (2nd difference) of the 300s
z-regime of the broad-book imbalance, weighted by the regime's level
extremity |z|. Mirrors the round-5 winning oir_zaccel_extreme_15s template
on a WIDER base that includes depth beyond level 5.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback for velocity and acceleration


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
    """d2z * |z| where d2z = 15s acceleration of the full-book-imbalance z.

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)       # 15s velocity of z
    d2z = dz - dz.shift(LAG)    # 15s acceleration of z
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted broad-book regime stretch: the 15s "
        "acceleration (2nd difference) of z_300(full-book imbalance), "
        "weighted by how stretched that regime is (|z|). The full-book "
        "imbalance includes depth BEYOND level 5 -- passive institutional "
        "queue placement and redemption-flow tilt that the top-5 engines "
        "(wdi, depth5_imb) cannot see. Its z-acceleration isolates "
        "INTENSIFYING one-sided broad-book repositioning from steady-state "
        "tilt: when the broad regime is already stretched (high |z|) and "
        "its rebuild is accelerating further (d2z pointing in the regime "
        "direction), large passive positions are being added or pulled at "
        "INCREASING speed -- a commitment signal that the broad tilt "
        "continues at 15-60s. Economically distinct from the round-4 "
        "fullbook_imb_zvel_div_15s (signed-difference of level minus "
        "velocity): that asks overextension-vs-fade; this asks whether the "
        "rate of change is ITSELF accelerating. A steady high-velocity "
        "broad regime scores ~0 here (constant dz -> d2z~0); only "
        "changing-velocity regimes fire."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6A family brief: z-acceleration-extremeness template "
        "(oir_zaccel_extreme_15s, round-5 strongest |t|) applied to the "
        "full-book imbalance base NOT yet covered in acceleration form."
    ),
    compute=compute,
)
