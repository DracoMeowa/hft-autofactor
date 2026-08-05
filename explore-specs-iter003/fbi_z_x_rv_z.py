"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

fbi_z_x_rv_z: full-book imbalance regime gated by the volatility regime
-- the patient-reservoir signal survives only when the market is quiet.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing windows


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(full-book imbalance, 300s) x z(rv_60s, 300s)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    fbi_z = _z(fbi, W)
    rv_z = _z(pl.col("rv_60s"), W)
    return part.select((fbi_z * rv_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fbi_z_x_rv_z",
    mechanism=(
        "Volatility-regime gate on the slow book state, with the QUIET-"
        "MARKET hypothesis. The full-book imbalance z (fullbook_imb_z_"
        "300s reconstruction) measures where patient positioning parks: "
        "the outer queue is institutional and creation/redemption "
        "inventory. Hypothesis: this reservoir signal has predictive "
        "content only in QUIET regimes, where prices trade off queue "
        "structure and patient orders get filled in order; in turbulent "
        "regimes exogenous shocks dominate, run over queue positioning, "
        "and the reservoir's information decays. The product fbi_z x "
        "rv_z therefore carries NEGATIVE IC: a bid-skewed whole book "
        "predicts appreciation only when rv_z < 0 (product negative, "
        "return positive), and symmetrically for asks. RV enters "
        "strictly as the CONDITIONING state -- its level/z as a "
        "predictor was IS-dead in round 1 and is not reused here. This "
        "is NOT the dead regime_vol_x_flow (rv z x TI z, round 1): that "
        "tested turbulence AMPLIFYING fast flow impact; this tests "
        "turbulence ERODING a slow book state -- the opposing "
        "falsifiable hypothesis on a different base."
    ),
    info_set="total_bid_vol, total_ask_vol, rv_60s",
    inspiration=(
        "iter-003 R3-D family brief direction 3 (fullbook imbalance "
        "conditioned on volatility regime: quiet vs turbulent); round-2 "
        "admitted fullbook_imb_z_300s; rv-as-condition-only per the "
        "round-1 dead-list carve-out."
    ),
    compute=compute,
)
