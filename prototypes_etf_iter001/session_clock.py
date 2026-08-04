"""ETF-structure candidate 6: intraday session-clock seasonality.

Every library factor ignores ts_ms, so time-of-day is an untouched GAP
dimension.  This prototype is the clean, honest test of PURE intraday
seasonality: it maps each snapshot to its elapsed continuous-trading time
(session clock in [0,1], 09:30->0, 11:30->0.5, 13:00->0.5, 15:00->1),
skipping the lunch break.  If 588000's short-horizon returns carry any
deterministic intraday drift (open/close auction effects, lunch-resumption
patterns, U-shape positioning), the session clock will show non-zero RankIC.
A null result here is itself informative: it says seasonality must enter as
CONDITIONING of other signals, not as a standalone directional factor.
"""
import polars as pl

#: session phase constants in ms-of-day (SSE continuous trading)
OPEN_MS = 34_200_000            # 09:30:00
MORNING_END_MS = 41_400_000     # 11:30:00
AFTERNOON_START_MS = 46_800_000  # 13:00:00
SESSION_LEN_MS = 14_400_000     # 4h of continuous trading


def _compute(part: pl.DataFrame) -> pl.Series:
    ts = pl.col("ts_ms").cast(pl.Float64)
    morning = ts - OPEN_MS
    afternoon = (MORNING_END_MS - OPEN_MS) + (ts - AFTERNOON_START_MS)
    clock = pl.when(ts <= MORNING_END_MS).then(morning).otherwise(afternoon)
    return part.select((clock / SESSION_LEN_MS).alias("value"))["value"]


PROTOTYPE = {
    "name": "session_clock",
    "mechanism": (
        "Intraday session-clock seasonality: maps each snapshot to its "
        "elapsed continuous-trading time in [0,1] (09:30->0, 11:30->0.5, "
        "13:00->0.5, 15:00->1), skipping the lunch break. This is the "
        "clean test of whether 588000 short-horizon returns carry any "
        "deterministic time-of-day drift (open/close effects, "
        "lunch-resumption, U-shape positioning). Pure calendar/time "
        "features have no microstructure state of their own, so a null "
        "result is informative: seasonality must enter as CONDITIONING of "
        "other signals, not as a standalone directional factor."
    ),
    "info_set": "ts_ms",
    "inspiration": (
        "digest: 'GAP dimension = time_of_day (ts_ms unused by all 12 "
        "factors - open/close auctions, lunch-break boundary, U-shape vol "
        "normalization, time-conditioning)'; docs/knowledge/02 section 6 "
        "intraday seasonality (U-shape volume/vol, 13:00 second open)."
    ),
    "compute": _compute,
}
