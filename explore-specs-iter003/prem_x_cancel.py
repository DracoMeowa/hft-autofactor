"""Explore-lane prototype spec (iter-003, etf-regime lens).

prem_x_cancel: z(iopv_premium, 100 rows) x z(cancel_ratio_60s, 100 rows).
Quote-pull activity DURING mispricing episodes: market makers protecting
themselves from being picked off by arbitrage flow.
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
    zv = _z("iopv_premium")
    zc = _z("cancel_ratio_60s")
    val = zv * zc
    return part.select(
        pl.when(zv.is_not_null() & zc.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="prem_x_cancel",
    mechanism=(
        "Defensive quote-pulls during mispricing precede the close: when "
        "the premium stretches, market makers face adverse selection from "
        "AP arbitrageurs who know the fair value, so they protect "
        "inventory by pulling quotes (cancel ratio rises) before "
        "re-quoting at adjusted prices. The JOINT episode -- stretched "
        "premium AND unusually high cancel activity relative to trailing "
        "300s -- marks the moment just before the re-quote: liquidity is "
        "thinning on the stale side and the next price move is in the "
        "arbitrage direction, closing the premium. The factor therefore "
        "predicts returns OPPOSITE to the premium's sign (negative IC "
        "under the reversion hypothesis): rich premium + defensive cancels "
        "-> down; deep discount + defensive cancels -> up. The partner "
        "column (cancel dynamics) is a dimension no iter-001 premium "
        "interaction touched (they used ofi/wdi flow legs, all dead)."
    ),
    info_set="iopv_premium, cancel_ratio_60s",
    inspiration=(
        "iter-003 etf-regime brief: quote-pull activity during mispricing "
        "episodes (market makers protecting themselves); quote withdrawal "
        "under adverse selection (Glosten & Milgrom 1985); iter-001 "
        "archive: prem_x_ofi / prem_x_wdi (flow-leg interactions) dead -- "
        "the cancel-dynamics leg is the untested conditioning column."
    ),
    compute=compute,
)
