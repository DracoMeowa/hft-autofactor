"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

iopv_drift_x_fbi: sustained NAV drift gated by the broad-book regime --
slow-x-slow: the arb trend converts into ETF drift only when the outer
queue accommodates it.
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
    """rolling_mean(iopv_velocity, 300s) x z(full-book imbalance, 300s)."""
    drift = pl.col("iopv_velocity").rolling_mean(window_size=W, min_samples=W)
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    fbi_z = _z(fbi, W)
    return part.select((drift * fbi_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_drift_x_fbi",
    mechanism=(
        "Slow-x-slow state interaction: sustained NAV drift "
        "(iopv_vel_drift_300s reconstruction, admitted at 900s) converts "
        "into ETF mid drift only when the BROAD BOOK accommodates the "
        "arbitrage trend. The outer queue aggregated by total_bid/ask_vol "
        "is where creation/redemption programs and institutional "
        "positioning park; a full-book regime leaning WITH the drift is "
        "sponsorship of the arb trend (program flow will keep working it) "
        "-> continuation amplified; a broad book leaning AGAINST the "
        "drift means opposing inventory absorbs the arb and the drift "
        "decays. Product drift x fbi_z carries POSITIVE IC at slow "
        "horizons (300-900s). Incremental to both parents: the drift "
        "parent averages over book state (a drift can persist in NAV "
        "while the ETF book refuses to follow), and the fbi_z parent "
        "averages over fundamental-anchor motion (a one-sided book with "
        "no anchor drift has nothing to track). Both legs are z/regime "
        "objects -- no dead level enters."
    ),
    info_set="iopv_velocity, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R3-D family brief direction 5 (iopv velocity family "
        "conditioned on book state); round-2 admitted iopv_vel_drift_300s "
        "x fullbook_imb_z_300s; slow-state pairing per the round-1 "
        "meta-lesson (300-900s reward regime states)."
    ),
    compute=compute,
)
