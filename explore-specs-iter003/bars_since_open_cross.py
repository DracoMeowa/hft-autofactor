"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

bars_since_open_cross: rows since mid last CROSSED the open price --
containment duration on the current side of the anchor (unsigned).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

MS_PER_ROW = 3000.0  # snapshots are 3s apart


def compute(part: pl.DataFrame) -> pl.Series:
    """Bars since the most recent open-recross; when no recross has
    happened yet, the containment spell is measured from the day's first
    snapshot (spring winding since the open)."""
    ts = pl.col("ts_ms").cast(pl.Float64)
    side = pl.when(pl.col("mid_px") > pl.col("open_px")).then(1.0).otherwise(-1.0)
    cross = pl.when(side != side.shift(1)).then(1.0).otherwise(0.0)
    # ts_ms strictly increases, so cum_min carries the first-row ts; the
    # clock starts at the last recross, or at the day's open if none yet.
    last_ts = pl.when(cross > 0.5).then(ts).otherwise(ts.cum_min()).cum_max()
    bars = (ts - last_ts) / MS_PER_ROW
    return part.select(bars.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="bars_since_open_cross",
    mechanism=(
        "Inventory-pressure clock on the anchored side. Each bar spent on "
        "one side of the open without recrossing, inventory, benchmark "
        "imbalances and one-sided hedges ACCUMULATE against the anchor: "
        "market makers defending quotes away from their reference price "
        "build skew, and creation/redemption desks stretch further from "
        "the open-referenced fair value. The longer the clock runs, the "
        "stronger the stored pressure for a re-crossing snap (the anchor "
        "acts as a spring being wound), and the more violent the eventual "
        "cross tends to be -- so long durations flag imminent two-sided "
        "risk and a reversion tilt, while short durations (fresh cross) "
        "mark the anchor just having been resolved, when the new side's "
        "directional flow typically has follow-through. Unlike its signed "
        "sibling (which asks WHICH consensus is holding), this factor "
        "asks HOW WOUND the spring is, independent of side -- a pure "
        "containment-duration state around the anchor."
    ),
    info_set="mid_px, open_px, ts_ms",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 3 (how long since "
        "mid last crossed the open); inventory-skew accumulation logic "
        "(Stoll 2003; Avellaneda & Stoikov 2008 quoting away from "
        "reference value), anchor-specific unlike the dead range-bound "
        "bars_since_extreme."
    ),
    compute=compute,
)
