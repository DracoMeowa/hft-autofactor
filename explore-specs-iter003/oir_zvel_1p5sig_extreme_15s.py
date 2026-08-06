"""Explore-lane prototype spec (iter-003 R6C, family R6C).

oir_zvel_1p5sig_extreme_15s: threshold sweep on the proven z-velocity-extremity
product. The 15s z-velocity of z_300(oir) weighted by extremity (dz * |z|),
but scored ONLY when |z| > 1.5 (top ~13% regime stretch). Tests whether the
touch-imbalance velocity signal begins accumulating BEFORE the 2-sigma
boundary -- if the looser 1.5sigma gate carries signal, the round-5 strict
2sigma gate is leaving moderate-stretch rows on the table.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback
THRESH = 1.5  # 1.5-sigma extremeness gate (looser than the proven 2sigma)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| when |z| > 1.5, else 0.0; warm-up rows null.

    The looser-extreme gate (1.5sigma) admits moderately-stretched
    touch-imbalance regimes into the velocity-extremity product.
    """
    z = _z(pl.col("oir"), W)
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
    name="oir_zvel_1p5sig_extreme_15s",
    mechanism=(
        "Loose-tail-isolated touch-imbalance velocity: the 15s z-velocity "
        "of z_300(oir) weighted by extremity (dz * |z|), but scored ONLY "
        "when the regime stretch exceeds 1.5 sigma (|z| > 1.5, top ~13% of "
        "the z distribution), zeroed otherwise. The top-of-book imbalance "
        "(oir) is the most actively managed quote slot; at 1.5sigma it is "
        "already a meaningful positioning tilt -- the heavy side of the "
        "best quote is pulling away from equilibrium beyond routine "
        "maintenance. Velocity of that moderate stretch (institutional "
        "repostioning of the touch) may carry informed-flow signal BEFORE "
        "the regime reaches the 2sigma extreme. The hypothesis under test "
        "is threshold placement: does the proven round-5 2sigma gate "
        "(oir/microprice/wdi zvel_2sig_extreme) discard useful signal in "
        "the 1.5-2sigma band, or is the 2sigma boundary genuinely where "
        "informed velocity separates from noise? Distinct from "
        "oir_zaccel_extreme_15s (acceleration, ungated) and from a "
        "hypothetical oir_zvel_2sig (this is looser): the 1.5sigma gate "
        "asks whether MODERATE stretch is enough for the velocity to be "
        "informed, not just the extreme tail."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6C family brief: threshold sweep on the proven "
        "z-velocity-extremity product; 1.5sigma is the loose end of the "
        "sweep, testing signal concentration below the 2sigma boundary."
    ),
    compute=compute,
)
