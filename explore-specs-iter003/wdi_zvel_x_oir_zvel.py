"""Explore-lane prototype spec (iter-003 R5, family R5-C).

wdi_zvel_x_oir_zvel: cross-surface extreme-velocity concordance -- the
product of the wdi extreme velocity (5-level depth) and the oir extreme
velocity (best-quote touch). Tests whether coordinated whole-visible-book
velocity predicts longer-horizon returns better than either surface alone.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(dz_wdi*|z_wdi|) * (dz_oir*|z_oir|); warm-up null."""
    zw = _z(pl.col("wdi"), W)
    dzw = zw - zw.shift(LAG)
    wdi_zvel = dzw * zw.abs()

    zo = _z(pl.col("oir"), W)
    dzo = zo - zo.shift(LAG)
    oir_zvel = dzo * zo.abs()

    return part.select((wdi_zvel * oir_zvel).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_x_oir_zvel",
    mechanism=(
        "Cross-surface velocity concordance: oir (best-quote touch "
        "imbalance) and wdi (5-level exp-decay depth imbalance) measure "
        "the SAME economic state at two different granularities of the "
        "visible book. oir is the fastest, cheapest signal -- the single "
        "queue pair where urgency first appears. wdi weights five levels "
        "and is slower but reflects broader queue commitment. When BOTH "
        "surfaces exhibit extreme-magnitude velocity in the SAME direction "
        "within 15s, the entire visible book from touch to level 5 is "
        "coordinated -- a rare high-conviction whole-surface move whose "
        "direction continues at 300-900s (the coordination across scales "
        "is the signature of a single informed meta-order, not idiosyncratic "
        "queue churn at one level). When they disagree (touch moving one "
        "way, depth the other), the surface is bifurcated -- one layer is "
        "reshuffling while another holds -- and the product cancels. The "
        "product is odd under sign flip of either surface, so it scores "
        "agreement magnitude, not a level. This asks a genuinely different "
        "question from either parent: not 'is wdi velocity extreme' or 'is "
        "oir velocity extreme' but 'are BOTH extreme in the same direction "
        "simultaneously' -- co-movement, not single-surface momentum."
    ),
    info_set="wdi, oir",
    inspiration=(
        "iter-003 R5-C family brief direction 5 (cross of TWO z-vel "
        "bases): when top-of-book (oir) and 5-level (wdi) velocity agree. "
        "Both are round-4 z-vel winners but on different book scales; "
        "their concordance tests a multi-surface coordination signal."
    ),
    compute=compute,
)
