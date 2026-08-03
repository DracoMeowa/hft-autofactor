"""Independent slow reference for snapshot-native factors (differential test C).

This module deliberately re-derives a subset of the canonical factors from the
RAW 3-second snapshot feed using plain NumPy/polars -- completely independent
of the C++ engine's book state -- so that the mask-validation stage can
differentially test the engine's snapshot-family outputs.

Supported factors (see docs/knowledge/microstructure_factors.md):
    quoted_spread_ticks | microprice_dev | oir | rv_60s | rv_300s | iopv_premium

Conventions (must match the engine exactly):
  * prices in CNY (float), tick = 0.001 CNY;
  * microprice_dev and iopv_premium expressed in basis points of the
    reference price;
  * rv_Hs = sqrt(sum of squared 3s log-mid returns) over the trailing H/3
    returns; a return is invalid across a snapshot gap > 6s (covers the lunch
    break and any missing snapshots) so windows never span a break;
  * warm-up / invalid inputs produce null (never zero-filled);
  * strictly causal: value at snapshot time t uses only snapshots <= t.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import polars as pl

TICK_CNY = 0.001
SNAPSHOT_GRID_MS = 3000
#: gaps strictly larger than this break realized-vol chains (2 grid periods)
MAX_SNAPSHOT_GAP_MS = 2 * SNAPSHOT_GRID_MS

SUPPORTED_FACTORS = (
    "quoted_spread_ticks",
    "microprice_dev",
    "oir",
    "rv_60s",
    "rv_300s",
    "iopv_premium",
)


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


def _rv_series(ts_ms: np.ndarray, mid: np.ndarray, window_returns: int) -> np.ndarray:
    """Causal rolling sqrt-of-sum-of-squared log-mid returns.

    Returns NaN until ``window_returns`` consecutive valid returns are
    available within an unbroken snapshot chain (gap <= MAX_SNAPSHOT_GAP_MS).
    """
    n = len(mid)
    out = np.full(n, np.nan)
    if n < 2:
        return out

    valid_mid = np.isfinite(mid)
    log_mid = np.where(valid_mid, np.log(np.where(valid_mid, mid, 1.0)), np.nan)

    r2 = np.full(n, np.nan)          # squared return ending at i
    chain = np.zeros(n, dtype=np.int64)  # consecutive valid returns up to i
    for i in range(1, n):
        gap_ok = ts_ms[i] - ts_ms[i - 1] <= MAX_SNAPSHOT_GAP_MS
        if gap_ok and valid_mid[i] and valid_mid[i - 1]:
            r = log_mid[i] - log_mid[i - 1]
            r2[i] = r * r
            chain[i] = chain[i - 1] + 1
        else:
            chain[i] = 0

    # cumulative sum of r2 inside each unbroken chain
    i = 1
    while i < n:
        if np.isnan(r2[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(r2[j]):
            j += 1
        seg = r2[i:j]
        cs = np.concatenate(([0.0], np.cumsum(seg)))
        for k in range(window_returns, len(seg) + 1):
            out[i + k - 1] = np.sqrt(cs[k] - cs[k - window_returns])
        i = j
    return out


def ref_snapshot_factors(snapshot_gz: Path, factor: str) -> pl.DataFrame:
    """Compute one snapshot-native factor directly from the raw snapshot feed.

    Returns a DataFrame with columns ``instrument, ts_ms, value`` (null for
    warm-up / invalid samples), sorted by (instrument, ts_ms).
    """
    if factor not in SUPPORTED_FACTORS:
        raise ValueError(
            f"unsupported reference factor {factor!r}; "
            f"supported: {', '.join(SUPPORTED_FACTORS)}"
        )

    df = _read_snapshot_csv(snapshot_gz)

    col_inst = _pick_column(df, "InstrumentID", "instrument")
    col_time = _pick_column(df, "UpdateTime", "DataTime", "ts_ms")
    col_bid1 = _pick_column(df, "BidPrice0", "bid1_px")
    col_ask1 = _pick_column(df, "AskPrice0", "ask1_px")
    col_bq1 = _pick_column(df, "BidVolume0", "bid1_qty")
    col_aq1 = _pick_column(df, "AskVolume0", "ask1_qty")
    col_last = _pick_column(df, "LastPrice", "last_px")
    col_iopv = _pick_column(df, "IOPV", "iopv")
    if col_inst is None or col_time is None:
        raise ValueError(f"snapshot file lacks InstrumentID/UpdateTime: {df.columns}")

    ts_list = [_parse_time_ms(v) for v in df[col_time].cast(pl.Utf8).to_list()]
    df = df.with_columns(pl.Series("ts_ms", ts_list, dtype=pl.Int64))
    df = df.filter(pl.col("ts_ms").is_not_null())

    out_frames: list[pl.DataFrame] = []
    for part in df.partition_by(col_inst):
        inst = str(part[col_inst][0])
        order = np.argsort(part["ts_ms"].to_numpy(), kind="mergesort")
        ts = part["ts_ms"].to_numpy()[order]

        def _f64(col: str | None, default: float = np.nan) -> np.ndarray:
            if col is None:
                return np.full(len(ts), default)
            arr = part[col].cast(pl.Float64).to_numpy()[order]
            return np.asarray(arr, dtype=np.float64)

        bid1, ask1 = _f64(col_bid1), _f64(col_ask1)
        bq1, aq1 = _f64(col_bq1), _f64(col_aq1)
        last = _f64(col_last)
        iopv = _f64(col_iopv)

        two_sided = (
            np.isfinite(bid1) & (bid1 > 0) & np.isfinite(ask1) & (ask1 > 0)
            & np.isfinite(bq1) & (bq1 > 0) & np.isfinite(aq1) & (aq1 > 0)
            & (ask1 >= bid1)
        )
        mid = np.where(two_sided, 0.5 * (bid1 + ask1), np.nan)

        if factor == "quoted_spread_ticks":
            value = np.where(two_sided, (ask1 - bid1) / TICK_CNY, np.nan)
        elif factor == "oir":
            value = np.where(two_sided, (bq1 - aq1) / (bq1 + aq1), np.nan)
        elif factor == "microprice_dev":
            micro = np.where(
                two_sided, (ask1 * bq1 + bid1 * aq1) / (bq1 + aq1), np.nan
            )
            value = np.where(
                two_sided & np.isfinite(mid) & (mid > 0),
                (micro - mid) / mid * 1.0e4,
                np.nan,
            )
        elif factor == "iopv_premium":
            iopv_ok = np.isfinite(iopv) & (iopv > 0)
            value = np.where(
                two_sided & iopv_ok & np.isfinite(last) & (last > 0),
                (last - iopv) / iopv * 1.0e4,
                np.nan,
            )
        elif factor in ("rv_60s", "rv_300s"):
            horizon_s = 60 if factor == "rv_60s" else 300
            window_returns = horizon_s // 3
            value = _rv_series(ts, mid, window_returns)
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
