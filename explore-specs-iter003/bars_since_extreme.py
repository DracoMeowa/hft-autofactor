"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

bars_since_extreme: rows since the last refresh of EITHER intraday extreme
(zero-sentinel cum_max trick). Duration of the current containment spell --
how long price has traded inside established day bounds.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

MS_PER_ROW = 3000.0  # snapshots are 3s apart


def compute(part: pl.DataFrame) -> pl.Series:
    """Rows since the most recent new-high-or-new-low event; null until the
    first boundary refresh of the day."""
    ts = pl.col("ts_ms").cast(pl.Float64)
    f = pl.when(
        (pl.col("high_px").diff(1) > 0.0) | (pl.col("low_px").diff(1) < 0.0)
    ).then(1.0).otherwise(0.0)
    # zero sentinel: ts_ms > 0 always, so cum_max carries the last event ts
    # forward through non-event rows (a null sentinel would leave gaps).
    last_either_ts = pl.when(f > 0.5).then(ts).otherwise(0.0).cum_max()
    bars = (ts - last_either_ts) / MS_PER_ROW
    out = (
        pl.when(f.cum_sum() >= 1.0)
        .then(bars)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="bars_since_extreme",
    mechanism=(
        "Duration of the current containment spell: how many 3s bars have "
        "passed since the day's high or low was last rewritten. Short "
        "values mark an active exploration regime (boundaries are being "
        "refreshed, information is arriving, recent drift is alive). Long "
        "values mark coiling: price pinned inside stale bounds while "
        "resting stops accumulate beyond both boundaries and market makers "
        "shrink inventory limits. Containment spells resolve abruptly -- "
        "and while the coil persists, the prior impulse decays, favoring "
        "mean reversion toward the middle of the range over continuation. "
        "Time-since-boundary is a regime-duration state absent from "
        "position, width, and migration constructions."
    ),
    info_set="high_px, low_px, ts_ms",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 4 (rows-since-refresh "
        "via the cum_max trick) applied to either boundary; range-bound "
        "duration / coil-before-resolution logic familiar from "
        "volatility-regime switching (Hamilton 1989 style states)."
    ),
    compute=compute,
)
