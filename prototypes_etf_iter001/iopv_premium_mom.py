"""ETF-structure candidate 1: IOPV premium momentum (rate of change).

The exchange IOPV updates on <=15s steps while the ETF trades continuously,
so the premium's RATE OF CHANGE carries lead-lag information distinct from
its LEVEL: a widening premium means the ETF is running ahead of fair value
(short-horizon continuation pressure), a narrowing premium means arbitrage
is already pulling it back.  The digest flags the IOPV dimension as THIN
(level only) and lists premium momentum/delta as unexplored and orthogonal
to the depth mega-family.
"""
import polars as pl

#: 10 x 3s snapshots = 30s lookback for the premium delta
DIFF_ROWS = 10


def _compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 30s change of the library iopv_premium; warm-up rows null."""
    return part.select(
        pl.col("iopv_premium").diff(DIFF_ROWS).alias("value")
    )["value"]


PROTOTYPE = {
    "name": "iopv_premium_mom",
    "mechanism": (
        "IOPV premium momentum: the exchange IOPV refreshes on <=15s steps "
        "while the ETF trades continuously, so the premium's 30s rate of "
        "change is distinct from its level. A widening premium (ETF running "
        "ahead of fair value) flags continuation pressure; a narrowing "
        "premium flags active arbitrage pull-back. This is the lead-lag "
        "between basket fair value and ETF price captured as premium "
        "velocity, not premium level."
    ),
    "info_set": "iopv_premium (library)",
    "inspiration": (
        "digest: 'IOPV dimension is thin (1 factor, level only); premium "
        "momentum/delta ... unexplored and orthogonal to the depth "
        "mega-family' (max |rho| of iopv_premium vs all others = 0.23); "
        "docs/knowledge/02 ETF lead-lag section (basket/fair-value pulls "
        "the ETF with seconds-to-minutes lag)."
    ),
    "compute": _compute,
}
