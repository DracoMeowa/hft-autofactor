"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

conc_imb_x_rv_z: depth-concentration regime gated by the volatility
regime -- placement URGENCY is amplified, not eroded, by turbulence
(the competing vol-gate hypothesis to fbi_z_x_rv_z).
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
    """z(bid/ask concentration asymmetry, 300s) x z(rv_60s, 300s)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    cb = pl.when(tb > 0.0).then(db / tb).otherwise(pl.lit(None, dtype=pl.Float64))
    ca = pl.when(ta > 0.0).then(da / ta).otherwise(pl.lit(None, dtype=pl.Float64))
    conc_z = _z(cb - ca, W)
    rv_z = _z(pl.col("rv_60s"), W)
    return part.select((conc_z * rv_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="conc_imb_x_rv_z",
    mechanism=(
        "Volatility-regime gate on the placement-style state, with the "
        "TURBULENCE-AMPLIFIES-URGENCY hypothesis -- the competing vol "
        "gate to fbi_z_x_rv_z. conc_imb_z_300s (admitted at 300/900s) "
        "measures head-heavy vs deep placement asymmetry: in CALM it is "
        "routine market-making posture with little directional content, "
        "but during turbulence packing one side's orders at the touch is "
        "aggressive queue-priority competition -- informed traders rush "
        "the head exactly when the price is moving, so head-heavy "
        "asymmetry in a high-rv regime marks genuine urgency. Hypothesis: "
        "the concentration signal is AMPLIFIED by rv_z: product conc_z x "
        "rv_z carries POSITIVE IC (head-heavy bid urgency in turbulence "
        "-> continued up; mirror for asks). Together the pair spans the "
        "two falsifiable answers to 'when does book structure matter': "
        "reservoir states decay in turbulence (fbi gate, negative IC) "
        "while urgency states feed on it (this gate, positive IC). RV is "
        "a conditioning state only (its predictor forms died round 1)."
    ),
    info_set="depth_bid5, depth_ask5, total_bid_vol, total_ask_vol, rv_60s",
    inspiration=(
        "iter-003 R3-D family brief direction 3 (conc_imb conditioned on "
        "volatility regime); round-2 admitted conc_imb_z_300s; "
        "competing-hypothesis design against fbi_z_x_rv_z within the "
        "same batch."
    ),
    compute=compute,
)
