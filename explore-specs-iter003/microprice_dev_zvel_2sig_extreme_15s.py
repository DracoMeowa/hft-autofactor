"""Explore-lane prototype spec (iter-003 R5, family R5-B).

microprice_dev_zvel_2sig_extreme_15s: NEW construction -- strict-extreme
gated velocity on microprice_dev. dz * |z| when |z| > 2.0, else 0.
Tests whether the microprice-deviation velocity signal concentrates in
the extreme tail of the deviation regime.
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
    """dz * |z| when |z| > 2.0, else 0.0; warm-up rows null."""
    z = _z(pl.col("microprice_dev"), W)
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
    name="microprice_dev_zvel_2sig_extreme_15s",
    mechanism=(
        "Tail-isolated microprice-deviation velocity: the 15s z-velocity "
        "of z_300(microprice_dev) weighted by extremity (dz * |z|), but "
        "scored ONLY when |z| > 2.0, zeroed otherwise. microprice_dev "
        "(micro minus mid) is the queue-weighted fair-value lead; beyond "
        "2 sigma the lead is genuinely stretched -- the heavy side of "
        "the queue is pulling the microprice hard away from mid, an "
        "extreme positioning edge. Velocity of that extreme (large "
        "institutional repositioning at the touch) continues at 15-60s; "
        "velocity near a neutral micro-deviation is quote noise. The "
        "strict gate isolates the ~5% highest-conviction rows, producing "
        "an event-sparse series distinct from the always-active round-4 "
        "microprice_dev_zvel_extreme_15s. The economic question is "
        "signal CONCENTRATION: does the micro-deviation velocity product "
        "live entirely in the tail?"
    ),
    info_set="microprice_dev",
    inspiration=(
        "iter-003 R5-B family brief: strict-extreme threshold variant of "
        "the round-4 product form on the microprice-deviation base."
    ),
    compute=compute,
)
