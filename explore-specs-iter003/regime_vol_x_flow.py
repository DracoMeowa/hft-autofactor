"""Explore-lane prototype spec (iter-003, etf-regime lens).

regime_vol_x_flow: z(rv_300s, 100 rows) x z(trade_imbalance_60s, 100 rows).
Flow impact CONDITIONED on the volatility regime -- the product of two
regime-relative states, not another flow accumulator.
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
    zv = _z("rv_300s")
    zf = _z("trade_imbalance_60s")
    val = zv * zf
    return part.select(
        pl.when(zv.is_not_null() & zf.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="regime_vol_x_flow",
    mechanism=(
        "Flow impact is regime-dependent: the same aggressive-flow "
        "imbalance moves price much more in a high-volatility regime "
        "(effective depth thins as market makers widen and step back, "
        "repricing is faster, and each unit of aggression consumes a "
        "larger fraction of the available queue) than in a quiet regime "
        "(deep books absorb flow with temporary impact). Z-scoring both "
        "legs over trailing 300s and multiplying up-weights signed flow "
        "exactly when its marginal impact is largest: unusual one-sided "
        "aggression during an unusual vol episode predicts a larger "
        "continuation move than the same aggression in calm. The factor "
        "is nearly orthogonal to the pure flow-accumulation cluster "
        "(ti_ewm_state / ti_accum / dd_flow): the vol-z multiplier is "
        "symmetric in flow sign, so the product decorrelates from any "
        "level/accumulator of imbalance."
    ),
    info_set="rv_300s, trade_imbalance_60s",
    inspiration=(
        "iter-003 etf-regime brief: flow impact conditioned on vol regime; "
        "iter-001/002 archive: the ti_ewm/ti_accum/dd_flow cluster is "
        "mutually corr 0.86-0.95 -- do not add another accumulator, "
        "condition instead. State-dependent price impact (impact grows as "
        "liquidity thins in vol regimes)."
    ),
    compute=compute,
)
