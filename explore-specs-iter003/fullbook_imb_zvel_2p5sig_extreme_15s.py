"""Explore-lane prototype spec (iter-003 R6C, family R6C).

fullbook_imb_zvel_2p5sig_extreme_15s: threshold sweep on the z-velocity-
extremity product applied to the FULL-BOOK imbalance. The 15s z-velocity
of z_300(full-book imbalance) weighted by extremity (dz * |z|), but scored
ONLY when |z| > 2.5. Tests the intermediate threshold on the WIDE book
that includes depth beyond level 5.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback
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
    """dz * |z| when |z| > 2.5, else 0.0; warm-up rows null.

    The 2.5sigma gate isolates the broad-book regime rows where hidden
    depth beyond level 5 is genuinely tilted.
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)
    return part.select(
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z.abs() > THRESH)
        .then(dz * z.abs())
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zvel_2p5sig_extreme_15s",
    mechanism=(
        "Tail-isolated broad-book velocity: the 15s z-velocity of "
        "z_300(full-book imbalance) weighted by extremity (dz * |z|), but "
        "scored ONLY when the regime stretch exceeds 2.5 sigma (|z| > 2.5, "
        "top ~1.2% of the z distribution), zeroed otherwise. The full-book "
        "imbalance includes depth BEYOND level 5 -- passive institutional "
        "queue placement and redemption-flow tilt invisible to the top-5 "
        "engines (wdi, depth5_imb). At 2.5sigma the broad book is in a "
        "committed one-sided state: large resting positions across the "
        "full visible + hidden depth. Velocity of that (institutional "
        "repositioning of the broad tilt) continues at 15-60s. The 2.5sigma "
        "gate sits between the proven 2sigma and the extreme 3sigma, "
        "testing whether the WIDE-book base (which is structurally noisier "
        "than the top-5 because it aggregates more levels) needs a tighter "
        "threshold to isolate informed velocity from full-book maintenance "
        "noise. Distinct from fullbook_imb_zvel_div_15s (level-minus-"
        "velocity divergence, ungated) and from top-5 zvel_2sig variants: "
        "different base, different threshold."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6C family brief: threshold sweep on the proven "
        "z-velocity-extremity product applied to the full-book imbalance "
        "base; 2.5sigma tests the intermediate threshold on the wide book."
    ),
    compute=compute,
)
