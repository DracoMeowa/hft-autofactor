"""Explore-lane prototype spec (iter-003 R6C, family R6C).

wdi_zvel_1p5sig_extreme_15s: threshold sweep on the proven z-velocity-
extremity product. The 15s z-velocity of z_300(wdi) weighted by extremity
(dz * |z|), scored ONLY when |z| > 1.5 (top ~13% stretch). Pairs with
wdi_zvel_3sig_extreme_15s to bracket the admitted wdi_zvel_2sig_extreme
on the SAME base from the loose side, directly answering 'does wdi
velocity signal live in the 1.5-2sigma band?'.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback
THRESH = 1.5  # 1.5-sigma extremeness gate (looser than the proven 2sigma)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| when |z| > 1.5, else 0.0; warm-up rows null.

    The loose-extreme gate (1.5sigma) admits moderately-stretched
    depth-imbalance regimes into the velocity-extremity product.
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
    name="wdi_zvel_1p5sig_extreme_15s",
    mechanism=(
        "Loose-tail-isolated depth-imbalance velocity: the 15s z-velocity "
        "of z_300(wdi) weighted by extremity (dz * |z|), but scored ONLY "
        "when the regime stretch exceeds 1.5 sigma (|z| > 1.5, top ~13% "
        "of the z distribution), zeroed otherwise. At 1.5sigma the 5-level "
        "depth imbalance is in a moderate positioning tilt -- beyond "
        "routine quote maintenance but below the 2sigma institutional "
        "crowding threshold. The hypothesis under test: does the "
        "depth-imbalance velocity signal (admitted at 2sigma in "
        "wdi_zvel_2sig_extreme_15s) begin accumulating in the 1.5-2sigma "
        "band? If the looser gate retains signal, the 2sigma boundary is "
        "too strict and moderate-stretch velocity also carries informed "
        "repositioning content. This spec and wdi_zvel_3sig_extreme_15s "
        "bracket the admitted 2sigma from both sides on the SAME wdi base, "
        "together forming a clean threshold sweep that isolates the "
        "threshold effect from the base effect."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6C family brief: threshold sweep on the proven "
        "z-velocity-extremity product; 1.5sigma on wdi is the loose end, "
        "pairing with wdi_zvel_3sig to bracket the admitted 2sigma."
    ),
    compute=compute,
)
