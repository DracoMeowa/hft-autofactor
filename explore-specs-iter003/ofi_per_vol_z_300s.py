"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ofi_per_vol_z_300s: book-flow surprise PER UNIT OF VOLATILITY RISK --
z_300(ofi_60s) divided by the relative rv_60s regime (clipped to [0.5, 2]).
Division form, rv_60s regime level, book channel.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s z window and vol-regime window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 300s) / clip(rv_60s / mean(rv_60s, 300s), 0.5, 2).

    Warm-up rows null; the regime denominator is guarded (null unless the
    trailing vol regime is positive) so no inf/NaN path exists.
    """
    z_ofi = _z(pl.col("ofi_60s"), W)
    rv = pl.col("rv_60s")
    rv_regime = rv.rolling_mean(window_size=W, min_samples=W)
    rel = (
        pl.when(rv_regime.is_not_null() & (rv_regime > 0.0))
        .then(rv / rv_regime)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    rel_c = rel.clip(lower_bound=0.5, upper_bound=2.0)
    return part.select((z_ofi / rel_c).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_per_vol_z_300s",
    mechanism=(
        "Flow intensity per unit of volatility risk: the same book-flow "
        "surprise means different things in different vol regimes. In a "
        "QUIET tape (rv_60s below its trailing-300s norm) a +2 z-ofi "
        "stands out -- one-sided queue building against a still book is "
        "cleanly informative and its repricing consequence is "
        "undiluted. In a VOL-BURST tape the identical flow z is partly "
        "reactive churn (inventory scrambling, hedging feedback), so its "
        "marginal information is discounted. Dividing the flow surprise "
        "by the clipped relative-vol regime (rv_60s / trailing mean, "
        "bounded to [0.5, 2] so the denominator can never explode or "
        "flip) amplifies flow exactly when the tape is calm enough for "
        "it to be trusted. This is the DIVISION form of flow-per-risk: "
        "the dead regime_vol_x_flow multiplied z(rv_300s) x z(ti_60s) "
        "-- different operation, different vol column, different "
        "channel -- and standalone RV z/ratio factors are dead; here "
        "volatility is only the normalizer of a flow surprise, never "
        "the signal itself."
    ),
    info_set="ofi_60s, rv_60s (library)",
    inspiration=(
        "iter-003 R3-C brief direction 3 (flow-per-risk normalization: "
        "ofi z divided by the rv_60s regime); state-dependent price "
        "impact (flow moves price more, and more cleanly, in calm "
        "regimes); regime_vol_x_flow rejection motivates the division "
        "form and the rv_60s regime level."
    ),
    compute=compute,
)
