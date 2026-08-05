"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

ofi_pos_frac_300s: fraction of the trailing 100 rows (300s) where ofi_60s
> 0 -- persistence of the LIMIT/PASSIVE-side flow direction, the book
channel's twin of ti_pos_frac_300s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing persistence window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing fraction of rows with ofi_60s > 0.

    Null inputs stay null (engine warm-up gaps excluded); warm-up
    (< 100 non-null rows) is null, never zero-filled.
    """
    x = pl.col("ofi_60s")
    ind = (
        pl.when(x.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(x > 0)
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select(
        ind.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_pos_frac_300s",
    mechanism=(
        "OFI aggregates limit-order book changes at the best quotes -- the "
        "PASSIVE channel: queue building on the bid vs the ask. A book "
        "whose order-flow imbalance is positive in (say) 90 of the last "
        "100 snapshots is being persistently rebuilt on the bid side for "
        "five minutes, which is what stealth accumulation looks like when "
        "informed agents prefer resting orders to aggression (they earn "
        "the spread instead of crossing it, especially on an ETF whose "
        "arbitrageurs quote actively). Persistence of positive OFI "
        "therefore predicts up-drift at 300-900s as the accumulated "
        "queues get consumed; the mirror predicts down-drift. This is the "
        "duration statistic of the book channel: magnitude OFI "
        "accumulators/momentum variants died OOS in round 1 "
        "(ofi_mom_60s, ofi_fast_slow), but the sign-frequency regime "
        "measure is a different object -- it ignores flow size and asks "
        "only how LONG the passive side has been leaning one way."
    ),
    info_set="ofi_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 3 (OFI analogue of the "
        "persistence fraction). OFI as the passive-channel information "
        "source (Cont, Kukanov & Stoikov 2014); round-1 lesson that OFI "
        "magnitude variants died while conditioned/derivative forms are "
        "the open lane."
    ),
    compute=compute,
)
