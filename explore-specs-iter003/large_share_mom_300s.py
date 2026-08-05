"""Explore-lane prototype spec (iter-003, etf-regime lens).

large_share_mom_300s: 100-row (300s) delta of large_trade_share_60s.  The
size distribution SHIFTING -- a regime-transition detector for institutional
participation, where the dead iter-002 form measured the level.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 100 x 3s rows = 300s momentum horizon for the size share
DIFF_ROWS = 100


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 300s change of the large-trade share; warm-up rows null."""
    return part.select(
        pl.col("large_trade_share_60s").diff(DIFF_ROWS).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="large_share_mom_300s",
    mechanism=(
        "Size-distribution momentum marks REGIME TRANSITIONS, not states: "
        "institutional participation ramps up and winds down over minutes "
        "(execution algorithms spread parent orders across time), so the "
        "5-minute CHANGE in the large-trade volume share detects the "
        "transition into an institutionally dominated episode (rising "
        "share) or back to retail/market-maker churn (falling share). "
        "Transitions persist -- a ramp is autocorrelated by construction "
        "of the execution schedules driving it -- giving the factor the "
        "slow state dynamics that match 300-900s horizons. On 588000 "
        "(STAR-50 ETF) the marginal institutional flow is creation/"
        "redemption-driven and trend-aligned with the just-revealed index "
        "move, so rising large-share episodes should drift with the "
        "contemporaneous trend. The level died in iter-002 (slow-state "
        "level = no edge); the delta is the meta-lesson-compliant object."
    ),
    info_set="large_trade_share_60s",
    inspiration=(
        "iter-002 archive: large_trade_share_level IC ~ 0; iter-003 "
        "etf-regime brief: 100-row delta of large-trade share = size "
        "distribution shifting; iter-001/002 meta-lesson: deltas/momenta "
        "of slow state survive where levels die. large_trade_share_60s "
        "materialized 2026-08-05 (wishlist)."
    ),
    compute=compute,
)
