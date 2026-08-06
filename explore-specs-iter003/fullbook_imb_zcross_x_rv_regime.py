"""Explore-lane prototype spec (iter-003 R5, family R5-C).

fullbook_imb_zcross_x_rv_regime: broad-book regime-flip velocity gated by
the SIGNED realized-variance regime (+1 turbulent / -1 calm). Tests whether
the direction-meaning of full-book repositioning flips between volatile
and quiet regimes.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z / regime window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """fullbook imb crossing velocity * signed_rv_regime(+1/-1); warm-up null."""
    z = _z(_fullbook_imb(), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    cross_vel = flip * (z - z_lag)
    rv = pl.col("rv_60s")
    rv_mean = rv.rolling_mean(window_size=W, min_samples=W)
    regime = (
        pl.when(rv_mean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(rv > rv_mean)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(-1.0))
    )
    return part.select((cross_vel * regime).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zcross_x_rv_regime",
    mechanism=(
        "Volatility-regime sign gate on broad-book flips: the full-book "
        "imbalance crossing (fullbook_imb_z_cross_vel_15s template) scores "
        "the broadest possible regime change -- the entire book from touch "
        "to tail relocating. The patient deeper-queue positioning this "
        "captures (institutional limit interest, ETF inventory) normally "
        "moves slowly, so a sign reversal inside 15s is either informed "
        "broad redeployment or forced unwinding. Hypothesis: in CALM "
        "regimes a deliberate broad relocation is credible informed "
        "repositioning and its direction continues (positive velocity IC); "
        "in TURBULENCE the same event is forced liquidation that "
        "OVERSHOOTS and mean-reverts (negative velocity IC). The signed "
        "+1/-1 regime indicator captures both in one coefficient: the "
        "product is the crossing velocity in calm and its opposite in "
        "turbulence. This is the signed dual-regime test on the BROADEST "
        "book base, complementing wdi_zvel_x_rv_regime (signed, "
        "CONTINUOUS velocity on 5-level depth) and the one-sided gates "
        "(specs 6-7). Each tests a different regime hypothesis on a "
        "different book scale."
    ),
    info_set="total_bid_vol, total_ask_vol, rv_60s",
    inspiration=(
        "iter-003 R5-C family brief direction 2: condition z-vel winners "
        "on rv regime; signed dual-regime on the fullbook imbalance "
        "crossing base, the broadest book scale."
    ),
    compute=compute,
)
