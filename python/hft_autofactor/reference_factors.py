"""Independent slow reference for snapshot-native factors (differential test C).

This module deliberately re-derives a subset of the canonical factors from the
RAW 3-second snapshot feed using plain NumPy/polars -- independent of the C++
engine's book state -- so that the mask-validation stage can differentially
test the engine's snapshot-family outputs.

Supported factors (see docs/knowledge/01-microstructure-factors.md):
    quoted_spread_ticks | microprice_dev | oir | rv_60s | rv_300s | iopv_premium

Engine-alignment contract (2026-08-04 alignment pass, see
docs/knowledge/04-lookahead-prevention.md §测试C):

  * Column lookup accepts BOTH header dialects: real SSE/SZSE dumps use
    bracketed level names (``BidPrice[0]``) while synthetic fixtures may use
    the plain form (``BidPrice0``); the C++ decoder accepts both, and so does
    this reference (the bracketed form is tried first).
  * prices in CNY (float), tick = 0.001 CNY;
  * microprice_dev and iopv_premium expressed in basis points of the
    reference price;
  * two-sided means PRICES only (bid1 > 0 and ask1 > 0), exactly like the
    engine's ``two_sided()``; oir/microprice additionally need bq1+aq1 > 0
    (missing volumes parse to 0, matching the engine's opt-int decoder);
    iopv_premium needs iopv > 0 (the engine's ``iopv_valid`` is iopv > 0);
  * rv_Hs matches ``RealizedVol`` in cpp/src/factors_snapshot.cpp EXACTLY:
    every snapshot contributes an entry (log-mid, or NaN when one-sided);
    the window is TIME-based (entries with t >= now - H*1000); squared
    log-returns are summed over ADJACENT valid entries inside the window --
    a NaN entry breaks adjacency around itself, but a time gap (missing
    snapshot) does NOT reset the chain; the value is emitted only once
    (now - first_snapshot_time) >= H*1000 AND the window holds at least 80%
    of the nominal return count ((H*1000/3000) * 4/5);
  * warm-up / invalid inputs produce null (never zero-filled);
  * strictly causal: value at snapshot time t uses only snapshots <= t.

Row-scope note: the engine only feeds ETF snapshots inside the continuous
session to its factors (is_etf_code + in_continuous_session in the engine
loop). Pass ``exchange="sse"|"szse"`` to reproduce that filter here; with
``exchange=None`` (default) every row is evaluated, which is what the
unit-test fixtures expect.

Residual simplifications vs the engine (documented, none material for the
supported factors): the engine skips a snapshot row entirely when LastPrice
is unparseable (hard decode error) while this reference keeps the row with a
null last price (only iopv_premium consumes it); the engine reads mid from
the snapshot-anchored book, which equals the snapshot's own top level after
apply_snapshot, so the raw top level is used here.
"""
from __future__ import annotations

import gzip
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

TICK_CNY = 0.001
SNAPSHOT_GRID_MS = 3000

SUPPORTED_FACTORS = (
    "quoted_spread_ticks",
    "microprice_dev",
    "oir",
    "rv_60s",
    "rv_300s",
    "iopv_premium",
)

#: continuous-session blocks per exchange (ms since midnight), mirroring
#: session_for() in cpp/src/session.cpp: [am_open, am_close) + [pm_open,
#: close_auction_start).  SSE funds have no closing auction (pm runs to
#: 15:00); SZSE continuous trading ends at 14:57.
_SESSION_BLOCKS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "sse": ((34_200_000, 41_400_000), (46_800_000, 54_000_000)),
    "szse": ((34_200_000, 41_400_000), (46_800_000, 53_820_000)),
}

#: ETF code prefixes per exchange, mirroring is_etf_code() in decode.cpp.
_ETF_PREFIXES: dict[str, tuple[str, ...]] = {
    "sse": ("50", "51", "52", "56", "58"),
    "szse": ("15", "16"),
}


def _read_snapshot_csv(snapshot_gz: Path) -> pl.DataFrame:
    """Read a gzip snapshot CSV, tolerating either direct or streamed decode.

    ``infer_schema_length=0`` keeps EVERY column as a string: UpdateTime
    values like ``093000000`` must not lose their leading zero to integer
    inference, and InstrumentIDs like ``510300`` must stay strings.
    """
    try:
        return pl.read_csv(
            snapshot_gz, null_values=["", "-"], infer_schema_length=0
        )
    except Exception:
        with gzip.open(snapshot_gz, "rb") as fh:
            return pl.read_csv(
                fh, null_values=["", "-"], infer_schema_length=0
            )


def _pick_column(df: pl.DataFrame, *candidates: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        hit = lower.get(cand.lower())
        if hit is not None:
            return hit
    return None


def _parse_time_ms(value: str | None) -> int | None:
    """Parse HHMMSSmmm / HH:MM:SS[.mmm] into ms since midnight.

    Digit strings of up to 9 chars are right-justified HHMMSSmmm: real
    SSE/SZSE dumps write the value as an integer with leading zeros dropped
    ("91400650" = 09:14:00.650). Consistent with the C++ engine and the
    mask-test parser.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if ":" in s:
            hh, mm, rest = s.split(":")
            sec = float(rest)
            return int(hh) * 3_600_000 + int(mm) * 60_000 + int(round(sec * 1000))
        if not s.isdigit() or len(s) > 9:
            return None
        v = int(s)
        ms = v % 1000; v //= 1000
        ss = v % 100;  v //= 100
        mm = v % 100;  v //= 100
        hh = v
        if hh > 23 or mm > 59 or ss > 59:
            return None
        return ((hh * 60 + mm) * 60 + ss) * 1000 + ms
    except (ValueError, IndexError):
        return None


def _rv_series(ts_ms: np.ndarray, mid: np.ndarray, window_s: int) -> np.ndarray:
    """Causal realized volatility matching the engine's RealizedVol exactly.

    Engine semantics (cpp/src/factors_snapshot.cpp): every snapshot pushes an
    entry (time, log-mid or NaN when one-sided); the trailing window is
    TIME-based (keep entries with t >= now - window_ms); squared log-returns
    are summed over adjacent valid entries inside the window, so a one-sided
    snapshot breaks adjacency around itself but a MISSING snapshot (time gap)
    does not reset anything.  Warm-up requires BOTH (now - first snapshot
    time) >= window_ms and >= 80% of the nominal return count present.
    """
    window_ms = window_s * 1000
    min_returns = (window_ms // 3000) * 4 // 5
    n = len(mid)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    entries: deque[tuple[int, float]] = deque()
    first_ts: int | None = None
    for i in range(n):
        t_i = int(ts_ms[i])
        if first_ts is None:
            first_ts = t_i
        m = mid[i]
        entries.append((t_i, float(np.log(m)) if np.isfinite(m) else float("nan")))
        while entries and entries[0][0] < t_i - window_ms:
            entries.popleft()

        total = 0.0
        ret_count = 0
        prev_valid = False
        prev_log = 0.0
        for _, lm in entries:
            if lm != lm:  # NaN: one-sided snapshot, break adjacency
                prev_valid = False
                continue
            if prev_valid:
                r = lm - prev_log
                total += r * r
                ret_count += 1
            prev_log = lm
            prev_valid = True

        if (t_i - first_ts) >= window_ms and ret_count >= min_returns:
            out[i] = np.sqrt(total)
    return out


def ref_snapshot_factors(
    snapshot_gz: Path,
    factor: str,
    *,
    exchange: str | None = None,
) -> pl.DataFrame:
    """Compute one snapshot-native factor directly from the raw snapshot feed.

    Returns a DataFrame with columns ``instrument, ts_ms, value`` (null for
    warm-up / invalid samples), sorted by (instrument, ts_ms).

    ``exchange`` (optional, "sse"|"szse"): reproduce the engine's row scope
    (ETF codes + continuous session only).  ``None`` evaluates every row.
    """
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(
            f"unsupported reference factor {factor!r}; "
            f"supported: {', '.join(SUPPORTED_FACTORS)}"
        )
    if exchange is not None and exchange not in _SESSION_BLOCKS:
        raise ValueError(
            f"unknown exchange {exchange!r} (use 'sse', 'szse' or None)"
        )

    df = _read_snapshot_csv(snapshot_gz)

    col_inst = _pick_column(df, "InstrumentID", "instrument")
    col_time = _pick_column(df, "UpdateTime", "DataTime", "ts_ms")
    # Real dumps use bracketed level names (BidPrice[0]); the plain form
    # (BidPrice0) is accepted too, matching the C++ decoder.
    col_bid1 = _pick_column(df, "BidPrice[0]", "BidPrice0", "bid1_px")
    col_ask1 = _pick_column(df, "AskPrice[0]", "AskPrice0", "ask1_px")
    col_bq1 = _pick_column(df, "BidVolume[0]", "BidVolume0", "bid1_qty")
    col_aq1 = _pick_column(df, "AskVolume[0]", "AskVolume0", "ask1_qty")
    col_last = _pick_column(df, "LastPrice", "Last", "last_px")
    col_iopv = _pick_column(df, "IOPV", "IOPVPrice", "iopv")
    if col_inst is None or col_time is None:
        raise ValueError(f"snapshot file lacks InstrumentID/UpdateTime: {df.columns}")

    ts_list = [_parse_time_ms(v) for v in df[col_time].cast(pl.Utf8).to_list()]
    df = df.with_columns(pl.Series("ts_ms", ts_list, dtype=pl.Int64))
    df = df.filter(pl.col("ts_ms").is_not_null())

    if exchange is not None:
        prefixes = _ETF_PREFIXES[exchange]
        (am_open, am_close), (pm_open, pm_close) = _SESSION_BLOCKS[exchange]
        is_etf = pl.lit(False)
        for p in prefixes:
            is_etf = is_etf | pl.col(col_inst).cast(pl.Utf8).str.starts_with(p)
        t = pl.col("ts_ms")
        in_session = ((t >= am_open) & (t < am_close)) | (
            (t >= pm_open) & (t < pm_close)
        )
        df = df.filter(is_etf & in_session)

    out_frames: list[pl.DataFrame] = []
    for part in df.partition_by(col_inst):
        inst = str(part[col_inst][0])
        order = np.argsort(part["ts_ms"].to_numpy(), kind="mergesort")
        ts = part["ts_ms"].to_numpy()[order]

        def _f64(col: str | None, default: float = np.nan) -> np.ndarray:
            if col is None:
                return np.full(len(ts), default)
            arr = np.asarray(
                part[col].cast(pl.Float64).to_numpy()[order], dtype=np.float64
            )
            # Missing cells take the decoder default, like the engine: volumes
            # default to 0 (opt-int), prices/last/iopv to NaN (=> invalid).
            if default == default:  # default is not NaN
                arr = np.where(np.isnan(arr), default, arr)
            return arr

        bid1, ask1 = _f64(col_bid1), _f64(col_ask1)
        # Missing volumes parse to 0 in the engine (opt-int decoder), not NaN.
        bq1, aq1 = _f64(col_bq1, 0.0), _f64(col_aq1, 0.0)
        last = _f64(col_last)
        iopv = _f64(col_iopv)

        # Engine two_sided(): PRICES only (bid > 0 and ask > 0).
        two_sided = (
            np.isfinite(bid1) & (bid1 > 0) & np.isfinite(ask1) & (ask1 > 0)
        )
        mid = np.where(two_sided, 0.5 * (bid1 + ask1), np.nan)

        if factor == "quoted_spread_ticks":
            value = np.where(two_sided, (ask1 - bid1) / TICK_CNY, np.nan)
        elif factor == "oir":
            denom_ok = two_sided & ((bq1 + aq1) > 0)
            value = np.where(denom_ok, (bq1 - aq1) / (bq1 + aq1), np.nan)
        elif factor == "microprice_dev":
            denom_ok = two_sided & ((bq1 + aq1) > 0)
            micro = np.where(
                denom_ok, (ask1 * bq1 + bid1 * aq1) / (bq1 + aq1), np.nan
            )
            value = np.where(
                denom_ok & np.isfinite(mid) & (mid > 0),
                (micro - mid) / mid * 1.0e4,
                np.nan,
            )
        elif factor == "iopv_premium":
            # engine: two_sided && iopv_valid (== iopv > 0) && last > 0
            iopv_ok = np.isfinite(iopv) & (iopv > 0)
            value = np.where(
                two_sided & iopv_ok & np.isfinite(last) & (last > 0),
                (last - iopv) / iopv * 1.0e4,
                np.nan,
            )
        elif factor in ("rv_60s", "rv_300s"):
            window_s = 60 if factor == "rv_60s" else 300
            value = _rv_series(ts, mid, window_s)
        else:  # pragma: no cover - guarded above
            raise AssertionError(factor)

        out_frames.append(
            pl.DataFrame(
                {
                    "instrument": [inst] * len(ts),
                    "ts_ms": ts,
                    "value": value,
                },
                schema={"instrument": pl.Utf8, "ts_ms": pl.Int64, "value": pl.Float64},
            )
        )

    if not out_frames:
        return pl.DataFrame(
            schema={"instrument": pl.Utf8, "ts_ms": pl.Int64, "value": pl.Float64}
        )
    return (
        pl.concat(out_frames)
        .with_columns(
            pl.when(pl.col("value").is_nan())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(pl.col("value"))
            .alias("value")
        )
        .sort(["instrument", "ts_ms"])
    )
