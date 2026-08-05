"""Explore-lane prototype spec (iter-003, etf-regime lens).

iopv_premium_mom_60s: 60s (20-row) delta of the IOPV premium.  Distinct
from the dead iter-001 iopv_premium_mom (30s / 10-row delta): the window is
doubled to span a full IOPV refresh + arbitrage-response cycle, justified
by the arbitrage-flow response speed below.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 20 x 3s rows = 60s premium delta window
DIFF_ROWS = 20


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 60s change of iopv_premium; warm-up rows null."""
    return part.select(
        pl.col("iopv_premium").diff(DIFF_ROWS).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="iopv_premium_mom_60s",
    mechanism=(
        "Premium velocity matched to the arbitrage response cycle: the "
        "exchange IOPV refreshes on <=15s steps, so a 30s premium delta "
        "(the dead iter-001 iopv_premium_mom) spans less than one full "
        "refresh-plus-response cycle and mostly measures IOPV-refresh noise "
        "and quote jitter. A 60s delta spans refresh -> detection -> "
        "arbitrage orders reaching the book, so it separates two regimes: "
        "premium still WIDENING after a full cycle means arbitrage flow has "
        "not yet dominated and the ETF keeps running ahead of fair value "
        "until that flow lands (price continuation in the premium's "
        "direction at 15-60s); premium NARROWING means arbitrage is winning "
        "and the just-revealed direction is being unwound. The object is "
        "the race between mispricing growth and arbitrage-flow response "
        "speed, observed on the race's own time scale."
    ),
    info_set="iopv_premium",
    inspiration=(
        "iter-001 archive: iopv_premium_mom (10-row/30s delta) IC ~ 0 -- "
        "the window sat inside the IOPV refresh cycle; iter-003 etf-regime "
        "brief: short-window premium delta justified via arbitrage-flow "
        "response speed (refresh <=15s + execution latency). Index/ETF "
        "lead-lag arbitrage dynamics; distinct window/justification from "
        "the dead form."
    ),
    compute=compute,
)
