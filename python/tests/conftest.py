"""Shared synthetic-data builders for the py-eval test-suite.

The interchange CSV schema (docs/interchange_format.md) is reproduced here so
tests never depend on the C++ engine binary.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

HORIZONS = (15, 30, 60, 300, 900)

BASE_HEADER = [
    "date", "exchange", "instrument", "ts_ms", "snap_seq", "flags",
    "mid_px", "last_px", "bid1_px", "ask1_px",
    "bid1_qty", "ask1_qty", "depth_bid5", "depth_ask5",
]


def factor_header(factors):
    return list(factors)


def label_header(horizons=HORIZONS):
    return (
        [f"fwd_mid_ret_{h}s" for h in horizons]
        + [f"fwd_last_ret_{h}s" for h in horizons]
    )


def full_header(factors, horizons=HORIZONS):
    return BASE_HEADER + factor_header(factors) + label_header(horizons)


def fmt(v):
    """Format a cell: None -> '' (NaN convention), floats at %.6f."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_interchange_csv(
    path: Path,
    *,
    date: str,
    exchange: str,
    rows: list[dict],
    factors,
    horizons=HORIZONS,
) -> Path:
    """Write rows to an interchange CSV.

    Each row dict needs keys: instrument, ts_ms, snap_seq, flags, mid_px,
    last_px, bid1_px, ask1_px, bid1_qty, ask1_qty, depth_bid5, depth_ask5,
    plus one key per factor and per label column (missing => empty cell).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = full_header(factors, horizons)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            out = [date, exchange]
            for col in header[2:]:
                out.append(fmt(r.get(col)))
            w.writerow(out)
    return path


def make_day_rows(
    instrument: str,
    *,
    n_snap: int,
    start_ms: int = 9 * 3_600_000 + 30 * 60_000,  # 09:30:00
    step_ms: int = 3000,
    factors=("oir",),
    horizons=HORIZONS,
    label_span_ok: bool = True,
):
    """Deterministic synthetic rows: factors = ts_ms/1e7, labels = future ts.

    label fwd_*_ret_{H}s = ts of the snapshot H seconds ahead (as a float),
    empty when beyond the last snapshot (ABSENT semantics) -- except when
    ``label_span_ok`` is False, which forces all labels empty.
    """
    rows = []
    for i in range(n_snap):
        ts = start_ms + i * step_ms
        r = {
            "instrument": instrument,
            "ts_ms": ts,
            "snap_seq": 1000 + i,
            "flags": 0,
            "mid_px": 4.0 + i * 0.001,
            "last_px": 4.0 + i * 0.001,
            "bid1_px": 3.999 + i * 0.001,
            "ask1_px": 4.001 + i * 0.001,
            "bid1_qty": 10000,
            "ask1_qty": 8000,
            "depth_bid5": 50000,
            "depth_ask5": 42000,
        }
        for f in factors:
            r[f] = ts / 1.0e7
        for h in horizons:
            ahead = i + (h * 1000) // step_ms
            value = None
            if label_span_ok and ahead < n_snap:
                value = float(start_ms + ahead * step_ms)
            r[f"fwd_mid_ret_{h}s"] = value
            r[f"fwd_last_ret_{h}s"] = value
        rows.append(r)
    return rows


def write_tick_gz(path: Path, *, n_rows: int = 200, start_ms: int = 34_200_000,
                  step_ms: int = 500, instrument: str = "510300") -> Path:
    """Minimal synthetic tick stream: SeqNo 1..n, TransactTime ascending."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ExchangeID", "ChannelNo", "SeqNo", "InstrumentID", "Trade2_Order1",
        "TradeDate", "TransactTime", "Price", "Volume", "TrdMoney", "OrdSide",
        "OrdType", "TrdBSFlag", "TrdBuyNo", "TrdSellNo", "OrdNo", "BizIndex",
        "TransFlag", "OrderTrdVolume", "TickStatus",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(header) + "\n")
        for seq in range(1, n_rows + 1):
            t = start_ms + seq * step_ms
            hh, rem = divmod(t, 3_600_000)
            mm, rem = divmod(rem, 60_000)
            ss, ms = divmod(rem, 1000)
            fields = [
                "111", "1", str(seq), instrument, "2", "20250603",
                f"{hh:02d}{mm:02d}{ss:02d}{ms:03d}",
                "4.001", "100", "400", "0", "", "B", str(seq), str(seq),
                str(seq), str(seq), "0", "100", "0",
            ]
            fh.write(",".join(fields) + "\n")
    return path


def write_snapshot_gz(path: Path, *, n_rows: int = 40,
                      start_ms: int = 34_200_000, step_ms: int = 3000,
                      instrument: str = "510300") -> Path:
    """Minimal synthetic snapshot stream with the columns the reference uses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "InstrumentID", "UpdateTime", "LastPrice", "PreClosePrice",
        "OpenPrice", "HighPrice", "LowPrice", "TradeVolume",
        "BidPrice0", "BidVolume0", "AskPrice0", "AskVolume0", "IOPV",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(n_rows):
            t = start_ms + i * step_ms
            hh, rem = divmod(t, 3_600_000)
            mm, rem = divmod(rem, 60_000)
            ss, ms = divmod(rem, 1000)
            ts = f"{hh:02d}{mm:02d}{ss:02d}{ms:03d}"
            fh.write(
                f"{instrument},{ts},4.001,4.000,4.000,4.010,3.990,{1000 * (i + 1)},"
                f"4.000,2000,4.002,1000,4.0008\n"
            )
    return path


@pytest.fixture
def small_cfg(tmp_path):
    """A PipelineConfig rooted in tmp_path with fake data roots."""
    from hft_autofactor.config import PipelineConfig

    sse_root = tmp_path / "data" / "sse"
    szse_root = tmp_path / "data" / "szse"
    sse_root.mkdir(parents=True)
    szse_root.mkdir(parents=True)
    cfg = PipelineConfig(
        data_roots={"sse": sse_root, "szse": szse_root},
        out_root=tmp_path / "factor_lzt",
        engine_bin=tmp_path / "bin" / "hftaf-engine",
        horizons_s=[15, 30, 60, 300, 900],
        factors=[],
        max_workers=1,
        commission_scenarios=["institutional", "retail_negotiated", "retail_default"],
    )
    cfg.ensure_dirs()
    return cfg
