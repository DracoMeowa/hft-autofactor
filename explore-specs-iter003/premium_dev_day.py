"""Explore-lane prototype spec (iter-003, etf-regime lens).

premium_dev_day: IOPV premium minus its EXPANDING intraday mean (today's own
norm).  Reference frame = the whole day so far, built causally from cum_sum
arithmetic -- no fixed window, precision grows as the day progresses.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: require >= 100 prior observations (~5 min) before the day-norm is trusted
MIN_OBS = 100


def compute(part: pl.DataFrame) -> pl.Series:
    """premium_i - expanding mean of premium over rows < i.

    Expanding mean via cum_sum arithmetic: non-null count and cumulative sum
    are both causal; the current row is excluded from its own baseline.
    Rows with fewer than MIN_OBS prior observations (and null-premium rows)
    are null, never zero-filled.
    """
    x = pl.col("iopv_premium")
    nonnull = x.is_not_null().cast(pl.Float64)
    cnt = nonnull.cum_sum()
    csum = x.cum_sum()
    prev_cnt = cnt - nonnull          # non-null observations strictly before i
    prev_sum = csum - x               # their sum (nulls skipped by cum_sum)
    denom = pl.when(prev_cnt >= MIN_OBS).then(prev_cnt).otherwise(pl.lit(1.0))
    dev = x - prev_sum / denom
    return part.select(
        pl.when(prev_cnt >= MIN_OBS)
        .then(dev)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="premium_dev_day",
    mechanism=(
        "Deviation from today's own premium norm: an ETF's premium level "
        "today is set by the day's fund-flow regime (net creation or "
        "redemption pressure can hold the premium away from zero for "
        "hours), and that structural level carries no short-horizon "
        "information. Subtracting the EXPANDING intraday mean removes "
        "today's norm without any fixed-window choice: what remains is the "
        "transient dislocation versus the day itself. Positive deviation = "
        "ETF temporarily rich vs where it has traded all day -> AP "
        "sell-ETF/buy-basket pressure and intraday mean reversion toward "
        "the day norm (negative forward returns after high values); the "
        "expanding baseline also gets more precise as the day ages, unlike "
        "a rolling window whose noise floor is constant. A different "
        "reference frame from the rolling-z siblings: 'unusual vs today' "
        "rather than 'unusual vs the recent minutes'."
    ),
    info_set="iopv_premium",
    inspiration=(
        "iter-003 etf-regime brief: premium relative to its own day is the "
        "untested transform after level/momentum and flow-interaction forms "
        "died in iter-001; expanding-mean via cum_sum arithmetic (polars "
        "has no causal expanding_mean in the safe idiom set). Intraday "
        "ETF premium/discount mean reversion toward NAV is the standard AP "
        "creation/redemption prediction."
    ),
    compute=compute,
)
