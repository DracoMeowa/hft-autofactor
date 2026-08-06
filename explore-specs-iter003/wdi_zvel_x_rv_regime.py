"""Explore-lane prototype spec (iter-003 R5, family R5-C).

wdi_zvel_x_rv_regime: depth-imbalance extreme velocity gated by the SIGNED
realized-variance regime (rv_60s above vs below its 300s rolling mean, mapped
to +1/-1). Tests whether the direction-meaning of depth-imbalance velocity
flips between turbulent and calm volatility regimes.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z / regime window
LAG = 5  # 5 x 3s rows = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(dz_wdi * |z_wdi|) * signed_rv_regime(+1/-1); warm-up null."""
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    zvel = dz * z.abs()
    rv = pl.col("rv_60s")
    rv_mean = rv.rolling_mean(window_size=W, min_samples=W)
    regime = (
        pl.when(rv_mean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(rv > rv_mean)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(-1.0))
    )
    return part.select((zvel * regime).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_x_rv_regime",
    mechanism=(
        "Volatility-regime sign gate on depth-imbalance velocity: "
        "wdi_zvel_extreme_15s (extremity-weighted 15s velocity of z_300(wdi)) "
        "measures how fast a crowded depth-imbalance regime is being "
        "rebuilt or abandoned. The SAME velocity event means opposite "
        "things in different volatility regimes. In TURBULENT regimes (rv "
        "above its trailing mean) informed traders rush the head of the "
        "queue exactly when price is moving -- an extreme depth-imbalance "
        "moving fast is aggressive urgency, and its direction CONTINUES. "
        "In CALM regimes (rv below mean) the same velocity is routine "
        "mean-reverting queue churn -- the depth tilt overshoots and "
        "Fades back. Mapping the regime to a signed +1/-1 indicator and "
        "multiplying captures both: the product's sign is the velocity "
        "direction in turbulence and its OPPOSITE in calm, so a single IC "
        "coefficient encodes both regimes at once. This is the SIGNED "
        "dual-regime test, distinct from the high-only / low-only gates "
        "(specs 6-7) which isolate one regime. RV enters strictly as a "
        "conditioning state -- its predictor forms died round 1."
    ),
    info_set="wdi, rv_60s",
    inspiration=(
        "iter-003 R5-C family brief direction 2: condition the z-vel "
        "winners on rv regime (rv_60s above/below rolling median) to test "
        "whether the signal only works in calm vs turbulent regimes; "
        "wdi_zvel_extreme_15s is the strongest z-vel winner."
    ),
    compute=compute,
)
