"""Explore-lane prototype spec (iter-003, etf-regime lens).

prem_reversion_x_rv: -z(iopv_premium, 100 rows) x z(rv_300s, 100 rows).
Premium reversion pressure conditioned on the volatility regime.  The dead
iter-001 interactions multiplied premium by FLOW columns (ofi/wdi); this one
conditions on VOLATILITY -- the regime that decides whether a premium spike
is a stale-IOPV artifact or a fundamental move.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 300s (100 x 3s rows) causal z-score windows for both legs
Z_WINDOW = 100


def _z(col: str) -> pl.Expr:
    """Causal rolling z-score; warm-up null, zero-variance windows -> 0.0."""
    x = pl.col(col)
    mean = x.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = x.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    z = (x - mean) / std
    return pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    zp = _z("iopv_premium")
    zr = _z("rv_300s")
    val = -zp * zr
    return part.select(
        pl.when(zp.is_not_null() & zr.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="prem_reversion_x_rv",
    mechanism=(
        "Volatility-regime-conditioned premium reversion: the exchange IOPV "
        "refreshes on <=15s steps, so during vol bursts the ETF price moves "
        "faster than its fair value can follow, and premium spikes in "
        "high-vol episodes are largely MECHANICAL (stale-denominator) "
        "dislocations rather than fundamental repricings. Mechanical "
        "dislocations close as the IOPV catches up and arbitrage flow "
        "lands, within one to five minutes. The factor takes the reversion "
        "direction (-z of the premium, so stretched-rich -> negative "
        "expected return) and scales it by the vol-regime z, concentrating "
        "the reversion bet exactly where the dislocation is most likely "
        "transient: positive values flag high-vol episodes with the premium "
        "stretched below norm (expect up-reversion), negative values flag "
        "high-vol premium-above-norm episodes (expect down-reversion). "
        "Outside vol regimes the factor is ~0 by construction."
    ),
    info_set="iopv_premium, rv_300s",
    inspiration=(
        "iter-001 archive: prem_x_ofi and prem_x_wdi (z(prem) x z(FLOW)) "
        "IC ~ 0 -- premium conditioned on flow is dead; iter-003 etf-"
        "regime brief: condition premium on the vol/liquidity REGIME "
        "instead. IOPV refresh latency <=15s (exchange methodology) makes "
        "fast-market premium spikes transient; rv_300s is the engine's "
        "unsigned realized variance."
    ),
    compute=compute,
)
