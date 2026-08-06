"""Explore-lane prototype spec (iter-003 R6C, family R6C).

fullbook_imb_zaccel_2p5sig_extreme_15s: threshold sweep on the z-ACCELERATION-
extremity product applied to the FULL-BOOK imbalance. The 15s z-acceleration
(2nd difference) of z_300(full-book imbalance) weighted by extremity
(d2z * |z|), scored ONLY when |z| > 2.5. Tests the 2.5sigma gate on the
WIDE book base whose hidden depth captures institutional flow invisible
at top-5.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s lookback for velocity and acceleration
THRESH = 2.5  # 2.5-sigma extremeness gate


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
    """d2z * |z| when |z| > 2.5, else 0.0; warm-up rows null.

    The 2.5sigma gate restricts the broad-book acceleration-extremity
    product to the highest-conviction rows.
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select(
        pl.when(z.is_null() | d2z.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z.abs() > THRESH)
        .then(d2z * z.abs())
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zaccel_2p5sig_extreme_15s",
    mechanism=(
        "Tail-isolated broad-book acceleration: the 15s z-acceleration of "
        "z_300(full-book imbalance) weighted by extremity (d2z * |z|), "
        "but scored ONLY when the regime stretch exceeds 2.5 sigma "
        "(|z| > 2.5, top ~1.2% of the z distribution), zeroed otherwise. "
        "The full-book imbalance includes hidden depth beyond level 5 -- "
        "passive institutional placement invisible to the top-5 engines. "
        "Its z-acceleration isolates INTENSIFYING broad-book repositioning "
        "(second derivative of the regime trajectory: rebuild speed that "
        "is itself accelerating). At 2.5sigma the broad regime is "
        "genuinely stretched: large resting positions across the full "
        "depth committed one way; acceleration there captures the moment "
        "that commitment is being added or withdrawn at increasing speed. "
        "Distinct from the ungated fullbook_imb_zaccel_extreme_15s "
        "(R6A, fires every row): the 2.5sigma gate tests whether the "
        "broad-book acceleration signal CONCENTRATES in the stretched "
        "tail. Distinct from top-5 zaccel_2sig variants: the wide-book "
        "base aggregates more depth levels and is structurally noisier, "
        "potentially requiring the tighter gate to isolate informed "
        "acceleration."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6C family brief: threshold sweep on the z-acceleration-"
        "extremity product on the full-book imbalance base; 2.5sigma tests "
        "the intermediate-threshold gate on the wide book."
    ),
    compute=compute,
)
