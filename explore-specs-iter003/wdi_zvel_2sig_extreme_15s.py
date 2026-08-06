"""Explore-lane prototype spec (iter-003 R5, family R5-B).

wdi_zvel_2sig_extreme_15s: NEW construction -- strict-extreme gated
velocity. The round-4 admitted wdi_zvel_extreme_15s (dz * |z|) fires on
every row; this variant gates it to rows where |z| > 2.0 (top ~5% regime
stretch), zeroing the rest. Tests whether the z-vs-velocity signal
CONCENTRATES in the extreme tails -- if the bulk-regime velocity is noise
and only the tail-regime velocity carries informed repositioning signal.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| when |z| > 2.0, else 0.0; warm-up rows null.

    The strict-extreme gate (|z| > 2) restricts the velocity-extremity
    product to the highest-conviction regime rows.
    """
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    return part.select(
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z.abs() > 2.0)
        .then(dz * z.abs())
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_2sig_extreme_15s",
    mechanism=(
        "Tail-isolated depth-imbalance velocity: the 15s z-velocity "
        "weighted by extremity (dz * |z|), but scored ONLY when the regime "
        "stretch exceeds 2 sigma (|z| > 2.0, top ~5% of the z "
        "distribution), zeroed otherwise. The hypothesis is that the "
        "round-4 admitted wdi_zvel_extreme_15s signal (dz * |z|, fires "
        "every row) is driven by its extreme tail: when the depth-"
        "imbalance regime is beyond 2 sigma, the queue state is genuinely "
        "crowded and ANY velocity is likely informed repositioning "
        "(institutional rebuild or abandonment of a deep multi-level "
        "position), while bulk-regime velocity (|z| < 2) is dominated by "
        "routine quote maintenance noise. The strict gate produces an "
        "event-sparse series (exact zero on ~95% of rows) whose nonzero "
        "support is the highest-conviction subset of the product form. "
        "Distinct from wdi_zvel_extreme_15s (always active): the threshold "
        "gate changes the economic question from 'velocity weighted by "
        "stretch' to 'velocity ONLY in the stretched tail'."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R5-B family brief: strict-extreme threshold variant of "
        "the round-4 winning product form; tests signal concentration in "
        "the 2-sigma tail."
    ),
    compute=compute,
)
