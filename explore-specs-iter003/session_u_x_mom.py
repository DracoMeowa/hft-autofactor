"""Explore-lane prototype spec (iter-003, etf-regime lens).

session_u_x_mom: NON-MONOTONIC U-shape session gate x 20-row mid momentum.
The iter-001 session_clock that passed weakly at 60s is a monotonic phase
clock; this is a different object: a U-shape window indicator (first/last
30 min of continuous trading) that CONDITIONS momentum instead of standing
alone.  Midday rows carry -momentum (reversal leg), not zero, so the factor
is a joint hypothesis: continuation in the U windows, reversal midday.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: session window edges in ms-of-day (SSE continuous trading, A-share)
OPEN_START_MS = 34_200_000     # 09:30:00
OPEN_U_END_MS = 36_000_000     # 10:00:00  (first 30 min)
CLOSE_U_START_MS = 52_200_000  # 14:30:00  (last 30 min)
CLOSE_END_MS = 54_000_000      # 15:00:00
#: 20 x 3s rows = 60s momentum horizon
MOM_ROWS = 20


def compute(part: pl.DataFrame) -> pl.Series:
    """+momentum in the U windows, -momentum midday; warm-up rows null."""
    ts = pl.col("ts_ms")
    u = ((ts >= OPEN_START_MS) & (ts < OPEN_U_END_MS)) | (
        (ts >= CLOSE_U_START_MS) & (ts < CLOSE_END_MS)
    )
    gate = pl.when(u).then(pl.lit(1.0)).otherwise(pl.lit(-1.0))
    mom = pl.col("mid_px").log().diff(MOM_ROWS)
    return part.select((gate * mom).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="session_u_x_mom",
    mechanism=(
        "Momentum is time-of-day conditioned: in the first and last 30 "
        "minutes of continuous trading, short-momentum is FLOW-informed -- "
        "overnight information digestion and auction-driven positioning at "
        "the open; index rebalancing, NAV-close and creation/redemption "
        "flows at the close -- and tends to continue over 15-60s. Midday, "
        "participation thins and short moves are dominated by noise and "
        "inventory jitter, which revert. The factor is +momentum inside "
        "the U windows and -momentum midday, a single signed object whose "
        "positive IC would confirm both legs (U-window continuation AND "
        "midday reversal). Different object from the iter-001 session_clock "
        "(monotonic phase, weak standalone 60s pass): this is a "
        "non-monotonic window indicator multiplied by fast state, i.e. "
        "seasonality entering as CONDITIONING of another signal -- exactly "
        "the route the session_clock post-mortem flagged as the only "
        "viable one."
    ),
    info_set="ts_ms, mid_px",
    inspiration=(
        "iter-001 archive: session_clock passed 60s weakly but is a pure "
        "monotonic clock (suspicious, no microstructure state); iter-003 "
        "etf-regime brief: a non-monotonic U-shape session feature is a "
        "legitimately different object. U-shape intraday volume/volatility "
        "and open/close positioning flows in A-share sessions; docs/"
        "knowledge/02 intraday seasonality."
    ),
    compute=compute,
)
