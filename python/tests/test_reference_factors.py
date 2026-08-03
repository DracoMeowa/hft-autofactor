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
    # 28 rows < 101 needed for a 100-return window -> all null
    assert all(v is None for v in vals.values())


def test_rv_chain_breaks_on_gap(tmp_path):
    """A missing snapshot (gap > 6s) must reset the rv window."""
    start = 34_200_000
    rows = []
    times = list(range(0, 10)) + list(range(20, 30))  # 30s hole after row 9
    for i, step in enumerate(times):
        ts = start + step * 3000
        rows.append(
            ("510300", _hhmmssmmm(ts), "4.000", "4.000", "4.000", "4.010",
             "3.990", 1000, "3.999", 1000, "4.001", 1000, "4.0000")
        )
    path = _write_snapshots(tmp_path / "gap.csv.gz", rows, HEADER)
    df = ref_snapshot_factors(path, "rv_60s")
    vals = _to_dict(df)
    # constant mid => rv == 0 where the 20-return window fits a solid chain
    tss = sorted(ts for (_, ts) in vals)
    # chain of 10 returns, then break, then 10 returns: never 20 in a row
    assert all(v is None for v in vals.values())
    assert len(tss) == 20


def test_unsupported_factor_raises(snap_file):
    with pytest.raises(ValueError):
        ref_snapshot_factors(snap_file, "wdi")
    assert "wdi" not in SUPPORTED_FACTORS


def test_empty_file_returns_empty(tmp_path):
    path = _write_snapshots(tmp_path / "empty.csv.gz", [], HEADER)
    df = ref_snapshot_factors(path, "oir")
    assert df.height == 0
    assert df.columns == ["instrument", "ts_ms", "value"]
