"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_side_hold_bars: signed duration of the current spell on one side of
the open -- bars held above (+) or below (-) since the last recross.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

MS_PER_ROW = 3000.0  # snapshots are 3s apart


def compute(part: pl.DataFrame) -> pl.Series:
    """Signed bars since the last open-recross; on days (or spells) with no
    recross yet, the spell is anchored to the day's first snapshot."""
    ts = pl.col("ts_ms").cast(pl.Float64)
    side = pl.when(pl.col("mid_px") > pl.col("open_px")).then(1.0).otherwise(-1.0)
    flip = pl.when(side != side.shift(1)).then(1.0).otherwise(0.0)
    # ts_ms strictly increases, so cum_min carries the first-row ts; the
    # spell starts at the last recross, or at the day's open if none yet.
    start_ts = pl.when(flip > 0.5).then(ts).otherwise(ts.cum_min()).cum_max()
    bars = (ts - start_ts) / MS_PER_ROW
    return part.select((bars * side).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="open_side_hold_bars",
    mechanism=(
        "Persistence of anchored acceptance, signed by side. The opening "
        "price is the day's reference bargain; HOW LONG mid has held one "
        "side of it without recrossing measures how durable the current "
        "value consensus is. A long unbroken spell above the open (+large) "
        "means every dip toward the anchor was bought for minutes at a "
        "time: inventory has migrated, weak hands are gone, and the "
        "anchored reversion pressure decays with spell length -- favoring "
        "continuation at short horizons. Short values (recent recross) "
        "mark the anchor as actively contested: two-sided flow, stop/"
        "benchmark orders clustered at the anchor being worked, and "
        "heightened two-way risk where mean reversion across the anchor "
        "dominates. This is a DURATION-of-SIGN state: it cannot be "
        "recovered from the deviation level (a slow drift and a choppy "
        "hold at the same deviation look identical in level), nor from "
        "momentum (spell length integrates many moves). When no recross "
        "has happened yet, the spell is measured from the day's first "
        "snapshot: acceptance-or-rejection running since the open itself."
    ),
    info_set="mid_px, open_px, ts_ms",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 3 (which side of "
        "the open mid holds and for how many bars); signed-duration "
        "upgrade of the round-2 bars-since-event construction (unsigned "
        "bars_since_extreme died: the sign is the information)."
    ),
    compute=compute,
)
