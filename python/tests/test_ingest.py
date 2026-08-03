"""Tests for ingest: job discovery, CSV->parquet build, panel loading."""
from __future__ import annotations

import gzip

import polars as pl
import pytest

from conftest import HORIZONS, make_day_rows, write_interchange_csv

from hft_autofactor.config import PipelineConfig
from hft_autofactor.ingest import (
    DEFAULT_FACTORS,
    build_day_parquet,
    discover_jobs,
    load_panel,
)

DATE = "20250603"


# --------------------------------------------------------------------- #
# discovery                                                             #
# --------------------------------------------------------------------- #
def _make_day_inputs(cfg: PipelineConfig, exchange: str, date: str, channels,
                     dump_suffix: str = "0603_081500"):
    day_dir = cfg.data_roots[exchange] / date[:6] / f"csv_{dump_suffix}"
    day_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(day_dir / "1_snapshot.csv.gz", "wt") as fh:
        fh.write("InstrumentID,UpdateTime\n510300,093000000\n")
    for ch in channels:
        with gzip.open(day_dir / f"1_channel_{ch}.csv.gz", "wt") as fh:
            fh.write("SeqNo,TransactTime\n1,093000000\n")
    return day_dir


def test_discover_jobs_basic(small_cfg):
    _make_day_inputs(small_cfg, "sse", DATE, channels=[1, 3])
    _make_day_inputs(small_cfg, "szse", DATE, channels=[2])

    jobs = discover_jobs(small_cfg, [DATE])
    keys = {(j.date, j.exchange, j.channel) for j in jobs}
    assert keys == {(DATE, "sse", 1), (DATE, "sse", 3), (DATE, "szse", 2)}

    job = next(j for j in jobs if j.exchange == "sse" and j.channel == 1)
    assert job.tick_gz.name == "1_channel_1.csv.gz"
    assert job.snapshot_gz.name == "1_snapshot.csv.gz"
    assert job.out_csv == small_cfg.raw_csv(DATE, "sse", 1)


def test_discover_jobs_skips_non_etf_channels(small_cfg):
    _make_day_inputs(small_cfg, "sse", DATE, channels=[1, 20, 801])
    jobs = discover_jobs(small_cfg, [DATE])
    assert [j.channel for j in jobs] == [1]


def test_discover_jobs_latest_dump_wins(small_cfg):
    _make_day_inputs(small_cfg, "sse", DATE, channels=[1], dump_suffix="0603_081500")
    _make_day_inputs(small_cfg, "sse", DATE, channels=[1], dump_suffix="0603_183000")
    jobs = discover_jobs(small_cfg, [DATE])
    assert len(jobs) == 1
    assert "csv_0603_183000" in str(jobs[0].tick_gz)


def test_discover_jobs_other_dates_ignored(small_cfg):
    _make_day_inputs(small_cfg, "sse", DATE, channels=[1])
    _make_day_inputs(small_cfg, "sse", "20250604", channels=[1],
                     dump_suffix="0604_081500")
    assert len(discover_jobs(small_cfg, [DATE])) == 1
    assert discover_jobs(small_cfg, ["20250101"]) == []


# --------------------------------------------------------------------- #
# parquet build                                                         #
# ---------------------------------------------------------------------
def _write_raw_day(cfg: PipelineConfig, date: str, per_channel: dict):
    """per_channel: {channel: [instrument, ...]}"""
    for ch, instruments in per_channel.items():
        rows = []
        for inst in instruments:
            rows.extend(make_day_rows(inst, n_snap=12, factors=("oir", "wdi")))
        path = cfg.raw_csv(date, "sse", ch)
        write_interchange_csv(
            path, date=date, exchange="sse", rows=rows, factors=("oir", "wdi"),
        )


def test_build_day_parquet_roundtrip(small_cfg):
    _write_raw_day(small_cfg, DATE, {1: ["510300", "510050"], 2: ["159915"]})

    out = build_day_parquet(DATE, small_cfg)
    assert out == small_cfg.parquet_path(DATE)
    assert out.is_file()

    df = pl.read_parquet(out)
    assert df.height == 3 * 12
    assert df.columns  # sanity
    assert set(df["instrument"].unique().to_list()) == {"510300", "510050", "159915"}
    # sorted by (instrument, ts_ms)
    pairs = list(zip(df["instrument"].to_list(), df["ts_ms"].to_list()))
    assert pairs == sorted(pairs)
    # channel column recovered from the file name
    assert set(df.filter(pl.col("instrument") == "159915")["channel"].unique().to_list()) == {2}
    # label columns survived with nulls at the day end (ABSENT semantics)
    assert df["fwd_mid_ret_900s"].null_count() > 0


def test_build_day_parquet_idempotent(small_cfg):
    _write_raw_day(small_cfg, DATE, {1: ["510300"]})
    p1 = build_day_parquet(DATE, small_cfg)
    mtime = p1.stat().st_mtime_ns
    p2 = build_day_parquet(DATE, small_cfg)
    assert p1 == p2
    assert p2.stat().st_mtime_ns == mtime  # skipped, not rewritten
    build_day_parquet(DATE, small_cfg, overwrite=True)
    assert p1.stat().st_mtime_ns != mtime


def test_build_day_parquet_rejects_instrument_in_two_channels(small_cfg):
    _write_raw_day(small_cfg, DATE, {1: ["510300"], 2: ["510300"]})
    with pytest.raises(ValueError, match="more than one channel"):
        build_day_parquet(DATE, small_cfg)


def test_build_day_parquet_rejects_duplicate_rows(small_cfg):
    _write_raw_day(small_cfg, DATE, {1: ["510300"]})
    # append an exact duplicate of the first row into the same channel file
    path = small_cfg.raw_csv(DATE, "sse", 1)
    rows = make_day_rows("510300", n_snap=12, factors=("oir", "wdi"))[:1]
    dup_file = small_cfg.out_root / "dup_stage.csv"
    write_interchange_csv(
        dup_file, date=DATE, exchange="sse", rows=rows,
        factors=("oir", "wdi"),
    )
    # merge the duplicate data row into the channel file
    header_plus = path.read_text().rstrip("\n") + "\n"
    dup_line = dup_file.read_text().splitlines()[1]
    dup_file.unlink()
    path.write_text(header_plus + dup_line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        build_day_parquet(DATE, small_cfg)


def test_build_day_parquet_warmup_column_not_string(small_cfg):
    """Regression: a factor column empty in the first >100 rows (warm-up) or
    empty all day (SSE cancel factors) must still come out Float64, not the
    String dtype polars' default 100-row inference would produce."""
    factors = ("warmfactor", "deadfactor")
    rows_a = make_day_rows("510050", n_snap=120, factors=factors)
    for r in rows_a:
        r["warmfactor"] = None  # first instrument: whole warm-up empty
    rows_b = make_day_rows("510300", n_snap=12, factors=factors)
    for r in rows_a + rows_b:
        r["deadfactor"] = None  # all-NaN column (SSE cancel decode)
    # 510050 sorts/writes first, so the first 120 rows of the file are all
    # empty in warmfactor -- squarely beyond the 100-row inference window.
    write_interchange_csv(
        small_cfg.raw_csv(DATE, "sse", 1),
        date=DATE, exchange="sse", rows=rows_a + rows_b, factors=factors,
    )

    out = build_day_parquet(DATE, small_cfg)
    df = pl.read_parquet(out)
    assert df["warmfactor"].dtype == pl.Float64
    assert df["deadfactor"].dtype == pl.Float64
    # real values survived the warm-up
    sub = df.filter(pl.col("instrument") == "510300")
    assert sub["warmfactor"].null_count() == 0
    assert df["warmfactor"].null_count() == 120


def test_build_day_parquet_missing_day_raises(small_cfg):
    with pytest.raises(FileNotFoundError):
        build_day_parquet("20250101", small_cfg)


# --------------------------------------------------------------------- #
# panel loading                                                         #
# ---------------------------------------------------------------------
def test_load_panel_filters(small_cfg):
    _write_raw_day(small_cfg, DATE, {1: ["510300", "510050"], 2: ["159915"]})
    build_day_parquet(DATE, small_cfg)

    panel = load_panel(small_cfg, [DATE])
    assert panel.height == 36
    for col in ("date", "exchange", "instrument", "ts_ms", "oir", "wdi",
                "fwd_mid_ret_15s", "fwd_last_ret_900s", "channel"):
        assert col in panel.columns

    sub = load_panel(small_cfg, [DATE], instruments=["159915"])
    assert set(sub["instrument"].unique().to_list()) == {"159915"}
    assert sub.height == 12

    only = load_panel(small_cfg, [DATE], factors=["oir"])
    assert "oir" in only.columns
    assert "wdi" not in only.columns
    assert "fwd_mid_ret_60s" in only.columns  # labels always kept


def test_load_panel_missing_partition_raises(small_cfg):
    with pytest.raises(FileNotFoundError):
        load_panel(small_cfg, ["20250101"])


def test_default_factors_constant_matches_spec():
    assert len(DEFAULT_FACTORS) == 12
    assert DEFAULT_FACTORS[:8] == (
        "quoted_spread_ticks", "microprice_dev", "oir", "wdi",
        "book_slope", "iopv_premium", "rv_60s", "rv_300s",
    )
    assert DEFAULT_FACTORS[8:] == (
        "ofi_60s", "trade_imbalance_60s", "order_arrival_60s", "cancel_ratio_60s",
    )
    assert len(HORIZONS) == 5
