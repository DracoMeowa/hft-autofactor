"""Explore-lane prototype spec (iter-003 R6C, family R6C).

wdi_zvel_3sig_extreme_15s: threshold sweep on the proven z-velocity-extremity
product. The 15s z-velocity of z_300(wdi) weighted by extremity (dz * |z|),
but scored ONLY when |z| > 3.0 (top ~0.3% stretch). Direct tighter-gate
sibling of the admitted wdi_zvel_2sig_extreme_15s: tests whether the
depth-imbalance velocity signal CONCENTRATES further in the 3sigma tail.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback
THRESH = 3.0  # 3-sigma extremeness gate (tighter than the proven 2sigma)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| when |z| > 3.0, else 0.0; warm-up rows null.

    The tight-extreme gate (3sigma) restricts the velocity-extremity
    product to the rarest depth-imbalance regime rows.
    """
    z = _z(pl.col("wdi"), W)
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
    name="wdi_zvel_3sig_extreme_15s",
    mechanism=(
        "Deep-tail-isolated depth-imbalance velocity: the 15s z-velocity "
        "of z_300(wdi) weighted by extremity (dz * |z|), but scored ONLY "
        "when the regime stretch exceeds 3 sigma (|z| > 3.0, top ~0.3% of "
        "the z distribution), zeroed otherwise. Beyond 3sigma the 5-level "
        "depth imbalance is in genuine institutional-crowding territory -- "
        "a multi-level queue state that only large committed positioning "
        "produces, well past the routine 2sigma institutional tilt. The "
        "hypothesis under test is signal CONCENTRATION at the extreme: the "
        "admitted wdi_zvel_2sig_extreme_15s fires on the top ~5% of rows; "
        "if the velocity-weighted-by-stretch signal is driven by its "
        "deepest tail, the 3sigma gate (top 0.3%) will retain the "
        "highest-conviction rows and the 2sigma gate is still partly "
        "contaminated by noise in the 2-3sigma band. Pairing with "
        "wdi_zvel_1p5sig_extreme_15s brackets the admitted 2sigma from "
        "BOTH sides on the same base, directly answering 'is 2sigma the "
        "right threshold for depth-imbalance velocity?'."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6C family brief: threshold sweep on the proven "
        "z-velocity-extremity product; 3sigma is the tight end of the "
        "sweep, testing whether signal concentrates further in the "
        "extreme tail."
    ),
    compute=compute,
)
