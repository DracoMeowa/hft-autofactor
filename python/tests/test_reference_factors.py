"""Differential-test reference: known-value checks for snapshot-native factors."""
from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np
import pytest

from hft_autofactor.reference_factors import (
    SUPPORTED_FACTORS,
    ref_snapshot_factors,
)


def _write_snapshots(path: Path, rows: list[tuple], header: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    return path


HEADER = [
    "InstrumentID", "UpdateTime", "LastPrice", "PreClosePrice",
    "OpenPrice", "HighPrice", "LowPrice", "TradeVolume",
    "BidPrice0", "BidVolume0", "AskPrice0", "AskVolume0", "IOPV",
]

# Real SSE/SZSE dumps use bracketed level names; the reference must accept
# both dialects (the C++ decoder does).
HEADER_BRACKETED = [
    "InstrumentID", "UpdateTime", "LastPrice", "PreClosePrice",
    "OpenPrice", "HighPrice", "LowPrice", "TradeVolume",
    "BidPrice[0]", "BidVolume[0]", "AskPrice[0]", "AskVolume[0]", "IOPV",
]


def _hhmmssmmm(ms: int) -> str:
    hh, rem = divmod(ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms3 = divmod(rem, 1000)
    return f"{hh:02d}{mm:02d}{ss:02d}{ms3:03d}"


@pytest.fixture
def snap_file(tmp_path):
    """25 two-sided snapshots at 3s cadence plus 3 one-sided rows at the end."""
    rows = []
    start = 9 * 3_600_000 + 30 * 60_000
    for i in range(25):
        ts = start + i * 3000
        # mid drifts as an exact geometric series: mid_i = 4 * exp(i * 1e-4)
        mid = 4.0 * math.exp(i * 1e-4)
        bid = round(mid - 0.001, 6)
        ask = round(mid + 0.001, 6)
        rows.append(
            ("510300", _hhmmssmmm(ts), f"{mid:.6f}", "4.000", "4.000",
             "4.010", "3.990", 1000 * (i + 1),
             f"{bid:.6f}", 2000, f"{ask:.6f}", 1000, "4.0008")
        )
    for i in range(3):
        ts = start + (25 + i) * 3000
        rows.append(
            ("510300", _hhmmssmmm(ts), "4.010", "4.000", "4.000", "4.010",
             "3.990", 26000, "0.000", 0, "4.012", 500, "4.0008")
        )
    return _write_snapshots(tmp_path / "1_snapshot.csv.gz", rows, HEADER)


def _to_dict(df):
    return {
        (inst, ts): val
        for inst, ts, val in zip(
            df["instrument"].to_list(), df["ts_ms"].to_list(), df["value"].to_list()
        )
    }


def test_quoted_spread_ticks(snap_file):
    df = ref_snapshot_factors(snap_file, "quoted_spread_ticks")
    vals = _to_dict(df)
    assert len(vals) == 28
    two_sided = [v for k, v in vals.items() if k[1] < 34_200_000 + 25 * 3000]
    assert all(abs(v - 2.0) < 1e-9 for v in two_sided)
    # one-sided rows -> null
    one_sided = [v for k, v in vals.items() if k[1] >= 34_200_000 + 25 * 3000]
    assert all(v is None for v in one_sided)


def test_oir(snap_file):
    df = ref_snapshot_factors(snap_file, "oir")
    vals = _to_dict(df)
    expected = (2000 - 1000) / (2000 + 1000)
    assert abs(vals[("510300", 34_200_000)] - expected) < 1e-12


def test_microprice_dev(snap_file):
    df = ref_snapshot_factors(snap_file, "microprice_dev")
    vals = _to_dict(df)
    bid, ask, bq, aq = 3.999, 4.001, 2000, 1000
    mid = (bid + ask) / 2
    micro = (ask * bq + bid * aq) / (bq + aq)
    expected_bps = (micro - mid) / mid * 1e4
    got = vals[("510300", 34_200_000)]
    assert abs(got - expected_bps) < 1e-6
    assert expected_bps > 0  # heavy bid queue pushes microprice up


def test_iopv_premium(snap_file):
    df = ref_snapshot_factors(snap_file, "iopv_premium")
    vals = _to_dict(df)
    last, iopv = 4.0, 4.0008  # first row: LastPrice = mid_0 = 4.000000
    expected = (last - iopv) / iopv * 1e4
    got = vals[("510300", 34_200_000)]
    assert abs(got - expected) < 1e-6
    assert got < 0  # discount


def test_rv_warmup_and_values(snap_file):
    df = ref_snapshot_factors(snap_file, "rv_60s")
    vals = _to_dict(df)
    tss = sorted(ts for (_, ts) in vals)
    # warm-up: first 20 values null (need 20 consecutive 3s returns)
    for ts in tss[:20]:
        assert vals[("510300", ts)] is None
    # thereafter: recompute naively from the mids exactly as written to file
    written = []
    for i in range(25):
        mid = 4.0 * math.exp(i * 1e-4)
        bid = round(mid - 0.001, 6)
        ask = round(mid + 0.001, 6)
        written.append((bid + ask) / 2)
    r2 = [
        (math.log(written[i]) - math.log(written[i - 1])) ** 2
        for i in range(1, 25)
    ]
    for idx in range(20, 25):
        # trailing 20 returns: those ending at snapshots idx-19 .. idx,
        # i.e. r2 indices idx-20 .. idx-1
        expected = math.sqrt(sum(r2[idx - 20 : idx]))
        got = vals[("510300", tss[idx])]
        assert got is not None
        assert abs(got - expected) < 1e-6, (idx, got, expected)


def test_rv300_needs_full_window(snap_file):
    df = ref_snapshot_factors(snap_file, "rv_300s")
    vals = _to_dict(df)
    # Warm-up needs elapsed >= 300s since the first snapshot AND >= 80% of
    # the nominal 100 returns in-window; 28 rows span only ~81s -> all null.
    assert all(v is None for v in vals.values())


def test_rv_gap_does_not_reset_window(tmp_path):
    """Engine semantics: a time gap does NOT reset the RV chain. Validity is
    gated by (elapsed >= window) AND (>= 80% of nominal returns present in
    the trailing TIME window). Rows 0..25 dense, gap 26..45, rows 46..66
    dense; the value at row 66 is taken over the trailing window only."""
    start = 34_200_000
    steps = list(range(0, 26)) + list(range(46, 67))
    rows = []
    mids = {}
    for step in steps:
        ts = start + step * 3000
        mid = 4.0 * math.exp(step * 1e-4)
        bid = round(mid - 0.001, 6)
        ask = round(mid + 0.001, 6)
        mids[step] = (bid + ask) / 2
        rows.append(
            ("510300", _hhmmssmmm(ts), f"{mid:.6f}", "4.000", "4.000",
             "4.010", "3.990", 1000, f"{bid:.6f}", 1000, f"{ask:.6f}", 1000, "4.0008")
        )
    path = _write_snapshots(tmp_path / "gap.csv.gz", rows, HEADER)
    df = ref_snapshot_factors(path, "rv_60s")
    vals = _to_dict(df)

    # Still warming up before 60s have elapsed since the first snapshot.
    assert vals[("510300", start + 10 * 3000)] is None

    # At step 66 the trailing 60s window holds rows 46..66 (21 rows, 20
    # returns >= 16) even though rows 26..45 are missing -> value emitted.
    got = vals[("510300", start + 66 * 3000)]
    assert got is not None
    r2 = [
        (math.log(mids[s]) - math.log(mids[s - 1])) ** 2 for s in range(47, 67)
    ]
    expected = math.sqrt(sum(r2))
    assert abs(got - expected) < 1e-6


def test_rv_one_sided_breaks_adjacency(tmp_path):
    """Engine semantics: a one-sided snapshot breaks the return chain around
    itself (no return is formed across it) but does not evict the rows on
    either side from the trailing window."""
    start = 34_200_000
    rows = []
    mids = {}
    for step in range(0, 41):
        ts = start + step * 3000
        if step == 20:
            # one-sided row: bid gone => mid invalid => breaks adjacency
            rows.append(
                ("510300", _hhmmssmmm(ts), "4.000", "4.000", "4.000", "4.010",
                 "3.990", 1000, "0.000", 0, "4.012", 500, "4.0008")
            )
            continue
        mid = 4.0 * math.exp(step * 1e-4)
        bid = round(mid - 0.001, 6)
        ask = round(mid + 0.001, 6)
        mids[step] = (bid + ask) / 2
        rows.append(
            ("510300", _hhmmssmmm(ts), f"{mid:.6f}", "4.000", "4.000",
             "4.010", "3.990", 1000, f"{bid:.6f}", 1000, f"{ask:.6f}", 1000, "4.0008")
        )
    path = _write_snapshots(tmp_path / "onesided.csv.gz", rows, HEADER)
    df = ref_snapshot_factors(path, "rv_60s")
    vals = _to_dict(df)

    # At step 40 the trailing window is rows 20..40; row 20 is one-sided, so
    # the returns are (21,22)..(39,40) -> 19 returns, skipping the pairs
    # (19,20) and (20,21) that would cross the one-sided snapshot.
    got = vals[("510300", start + 40 * 3000)]
    assert got is not None
    r2 = [
        (math.log(mids[s]) - math.log(mids[s - 1])) ** 2 for s in range(22, 41)
    ]
    expected = math.sqrt(sum(r2))
    assert abs(got - expected) < 1e-6


def test_bracketed_header_columns(tmp_path):
    """Real dumps use bracketed level names (BidPrice[0]); the reference must
    read them exactly like the C++ decoder does."""
    start = 34_200_000
    rows = []
    for i in range(5):
        ts = start + i * 3000
        rows.append(
            ("510300", _hhmmssmmm(ts), "4.000", "4.000", "4.000", "4.010",
             "3.990", 1000, "3.999", 2000, "4.001", 1000, "4.0008")
        )
    path = _write_snapshots(tmp_path / "bracketed.csv.gz", rows, HEADER_BRACKETED)

    vals = _to_dict(ref_snapshot_factors(path, "oir"))
    assert abs(vals[("510300", start)] - (2000 - 1000) / 3000) < 1e-12
    vals = _to_dict(ref_snapshot_factors(path, "quoted_spread_ticks"))
    assert abs(vals[("510300", start)] - 2.0) < 1e-9


def test_exchange_scope_filter(tmp_path):
    """exchange=... reproduces the engine row scope: ETF codes only, and only
    the continuous session (SSE pm runs to 15:00, SZSE stops at 14:57)."""
    con = 10 * 3_600_000                 # 10:00:00 continuous
    auc = 9 * 3_600_000 + 20 * 60_000    # 09:20:00 opening auction
    szse_pre = 14 * 3_600_000 + 56 * 60_000     # 14:56:00
    szse_post = 14 * 3_600_000 + 57 * 60_000 + 30_000  # 14:57:30 auction
    row = lambda inst, ts: (
        inst, _hhmmssmmm(ts), "4.000", "4.000", "4.000", "4.010",
        "3.990", 1000, "3.999", 2000, "4.001", 1000, "4.0008",
    )
    rows = [
        row("510300", auc),     # SSE ETF, auction time -> dropped
        row("510300", con),     # SSE ETF, continuous  -> kept by sse
        row("600519", con),     # SSE stock (not ETF)  -> dropped
        row("159915", con),     # SZSE ETF             -> kept by szse only
        row("159915", szse_pre),
        row("159915", szse_post),  # SZSE closing auction -> dropped by szse
    ]
    path = _write_snapshots(tmp_path / "scope.csv.gz", rows, HEADER)

    got = set(_to_dict(ref_snapshot_factors(path, "oir", exchange="sse")))
    assert got == {("510300", con)}

    got = set(_to_dict(ref_snapshot_factors(path, "oir", exchange="szse")))
    assert got == {("159915", con), ("159915", szse_pre)}

    # Without a scope every row is evaluated (fixture/unit-test default).
    got = set(_to_dict(ref_snapshot_factors(path, "oir")))
    assert len(got) == 6


def test_missing_volume_parses_as_zero(tmp_path):
    """Engine opt-int decoder: an empty volume field parses as 0 (not NaN)."""
    start = 34_200_000
    rows = [
        ("510300", _hhmmssmmm(start), "4.000", "4.000", "4.000", "4.010",
         "3.990", 1000, "3.999", "", "4.001", 1000, "4.0008"),
    ]
    path = _write_snapshots(tmp_path / "novol.csv.gz", rows, HEADER)
    vals = _to_dict(ref_snapshot_factors(path, "oir"))
    assert abs(vals[("510300", start)] - (0 - 1000) / (0 + 1000)) < 1e-12


def test_unsupported_factor_raises(snap_file):
    with pytest.raises(ValueError):
        ref_snapshot_factors(snap_file, "wdi")
    assert "wdi" not in SUPPORTED_FACTORS


def test_empty_file_returns_empty(tmp_path):
    path = _write_snapshots(tmp_path / "empty.csv.gz", [], HEADER)
    df = ref_snapshot_factors(path, "oir")
    assert df.height == 0
    assert df.columns == ["instrument", "ts_ms", "value"]
