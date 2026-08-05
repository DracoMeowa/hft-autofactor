"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

extreme_recency_asym: bounded relative recency of the day's two boundaries
(zero-sentinel cum_max trick): (since_last_new_low - since_last_new_high) /
(since_last_new_low + since_last_new_high), in (0,1) when the high is the
fresher boundary, (-1,0) when the low is. A side never refreshed counts as
stale since the session start; the ratio decays toward 0 while both bounds
go untouched.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-9
MS_PER_ROW = 3000.0  # snapshots are 3s apart


def compute(part: pl.DataFrame) -> pl.Series:
    """Bounded boundary-recency asymmetry; null until at least one boundary
    type has been touched."""
    ts = pl.col("ts_ms").cast(pl.Float64)
    f_hi = pl.when(pl.col("high_px").diff(1) > 0.0).then(1.0).otherwise(0.0)
    f_lo = pl.when(pl.col("low_px").diff(1) < 0.0).then(1.0).otherwise(0.0)
    # zero sentinel: ts_ms > 0 always, so cum_max carries the last event ts
    # forward through non-event rows (a null sentinel would leave gaps).
    last_hi_ts = pl.when(f_hi > 0.5).then(ts).otherwise(0.0).cum_max()
    last_lo_ts = pl.when(f_lo > 0.5).then(ts).otherwise(0.0).cum_max()
    since_hi = ts - last_hi_ts
    since_lo = ts - last_lo_ts
    denom = since_hi + since_lo
    asym = (since_lo - since_hi) / denom
    seen = f_hi.cum_sum() + f_lo.cum_sum()
    out = (
        pl.when((seen >= 1.0) & (denom > EPS))
        .then(asym)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="extreme_recency_asym",
    mechanism=(
        "Which day boundary was touched more recently, in bounded relative "
        "units: positive when the day-HIGH is the fresher extremum event, "
        "negative when the day-LOW is. The most recent boundary print "
        "defines the market's current exploration direction: a session "
        "whose latest extremum is a new high is in upside breakout mode -- "
        "resting buy-stops above the high get triggered, attention and "
        "momentum algos pile in, and short-horizon drift tends to follow "
        "the fresher boundary; the mirror holds for fresh lows. The "
        "relative (ratio) form decays toward neutral while BOTH bounds go "
        "untouched -- recency information genuinely ages -- and a side "
        "never refreshed at all counts as maximally stale (session start). "
        "Pure event-recency state: unlike range position it ignores where "
        "mid sits inside the envelope, and unlike range width it ignores "
        "how much envelope exists -- only WHICH side last won."
    ),
    info_set="high_px, low_px, ts_ms",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 4 (rows-since-refresh "
        "variant via the cum_max trick) on the batch-2 feed extremes; "
        "event-recency conditioning in the extremum-anchor family of the "
        "round-1 range-position champion."
    ),
    compute=compute,
)
