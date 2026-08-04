"""Mask-validation logic tests: truncation, compare_prefix, full day flow.

The engine is MOCKED with a tiny deterministic Python "engine" that is
prefix-consistent by construction (factor value = f(ts)) and, with
--canaries, emits a deliberately look-ahead column.  This proves the whole
mask_test_day orchestration detects leakage without needing the C++ binary.
"""
from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import pytest

from conftest import (
    HORIZONS,
    full_header,
    make_day_rows,
    write_interchange_csv,
    write_snapshot_gz,
    write_tick_gz,
)

from hft_autofactor.validation import mask_test as mt
from hft_autofactor.validation.mask_test import (
    TruncationPoint,
    choose_truncation_points,
    compare_prefix,
    engine_cli_args,
    parse_time_ms,
    truncate_snapshot_file,
    truncate_tick_file,
)

FACTORS = ("oir", "microprice_dev")
DATE = "20250603"
START_MS = 34_200_000  # 09:30:00
N_SNAP = 50
CUT_IDX = 30
CUT_MS = START_MS + CUT_IDX * 3000


# --------------------------------------------------------------------- #
# fixtures: full CSV + truncated variants                               #
# --------------------------------------------------------------------- #
def _write_full(path: Path, n_snap: int = N_SNAP) -> Path:
    rows = make_day_rows("510300", n_snap=n_snap, start_ms=START_MS,
                         factors=FACTORS)
    return write_interchange_csv(path, date=DATE, exchange="sse", rows=rows,
                                 factors=FACTORS)


def _write_trunc(path: Path, cut_idx: int = CUT_IDX) -> Path:
    """Truncated-run output: rows 0..cut_idx with labels recomputed on the
    shorter horizon (values beyond the last truncated snapshot are ABSENT)."""
    rows = make_day_rows("510300", n_snap=cut_idx + 1, start_ms=START_MS,
                         factors=FACTORS)
    return write_interchange_csv(path, date=DATE, exchange="sse", rows=rows,
                                 factors=FACTORS)


@pytest.fixture
def full_csv(tmp_path):
    return _write_full(tmp_path / "full.csv")


@pytest.fixture
def trunc_csv(tmp_path):
    return _write_trunc(tmp_path / "trunc.csv")


# --------------------------------------------------------------------- #
# compare_prefix                                                        #
# --------------------------------------------------------------------- #
def test_compare_prefix_identical(full_csv, trunc_csv):
    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=900)
    assert diff.identical is True
    assert diff.first_diff is None
    assert diff.n_rows_full == CUT_IDX + 1
    assert diff.n_rows_trunc == CUT_IDX + 1


def test_compare_prefix_factor_tamper_detected(full_csv, trunc_csv):
    text = trunc_csv.read_text().splitlines()
    header = text[0].split(",")
    oir_idx = header.index("oir")
    fields = text[11].split(",")  # row idx 10 -> ts <= cut
    fields[oir_idx] = "9.999999"
    text[11] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")

    diff = compare_prefix(full_csv, trunc_csv, CUT_MS)
    assert diff.identical is False
    assert "oir" in diff.first_diff
    assert "instrument=510300" in diff.first_diff


def test_label_change_inside_label_horizon_is_allowed(full_csv, trunc_csv):
    """Labels at t in (cut - H_max, cut] may legitimately differ."""
    text = trunc_csv.read_text().splitlines()
    header = text[0].split(",")
    lab_idx = header.index("fwd_mid_ret_15s")
    row_idx = CUT_IDX  # ts == cut -> certainly within (cut-900s, cut]
    fields = text[1 + row_idx].split(",")
    fields[lab_idx] = "" if fields[lab_idx] else "1.000000"
    text[1 + row_idx] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")

    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=900)
    assert diff.identical is True


def test_label_change_before_label_cutoff_detected(full_csv, trunc_csv):
    """With a shorter max horizon the same tamper IS in compare scope."""
    text = trunc_csv.read_text().splitlines()
    header = text[0].split(",")
    lab_idx = header.index("fwd_mid_ret_15s")
    row_idx = 10  # ts = START+30s <= CUT_MS - 15s
    fields = text[1 + row_idx].split(",")
    fields[lab_idx] = "0.123456"
    text[1 + row_idx] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")

    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=15)
    assert diff.identical is False
    assert "fwd_mid_ret_15s" in diff.first_diff


def test_label_absent_in_truncated_run_inside_horizon_is_allowed(full_csv, trunc_csv):
    """full-present/trunc-absent inside the label horizon is legitimate.

    A label resolves at the FIRST snapshot U >= t+H; when the instrument has
    snapshot gaps, U can fall after the cut, so the truncated run cannot
    resolve a label the full run resolves.  (Real-data case: sparse 501xxx
    LOFs during the 20250701 smoke run.)
    """
    text = trunc_csv.read_text().splitlines()
    header = text[0].split(",")
    lab_idx = header.index("fwd_mid_ret_15s")
    row_idx = 10  # ts = START+30s <= CUT_MS - 15s -> inside compare scope
    fields = text[1 + row_idx].split(",")
    assert fields[lab_idx] != ""  # sanity: full run resolved this label
    fields[lab_idx] = ""
    text[1 + row_idx] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")

    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=15)
    assert diff.identical is True


def test_label_present_in_truncated_but_absent_in_full_detected(full_csv, trunc_csv):
    """The converse direction stays enforced: a truncated-present label must
    equal the full run's cell, so trunc-present/full-absent is a mismatch."""
    text = trunc_csv.read_text().splitlines()
    header = text[0].split(",")
    lab_idx = header.index("fwd_mid_ret_15s")
    row_idx = 10
    fields = text[1 + row_idx].split(",")
    fields[lab_idx] = "0.500000"
    text[1 + row_idx] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")

    ftext = full_csv.read_text().splitlines()
    ffields = ftext[1 + row_idx].split(",")
    ffields[lab_idx] = ""
    ftext[1 + row_idx] = ",".join(ffields)
    full_csv.write_text("\n".join(ftext) + "\n")

    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=15)
    assert diff.identical is False
    assert "fwd_mid_ret_15s" in diff.first_diff


def test_compare_prefix_boundary_row_nonlabel_diff_is_allowed(full_csv, trunc_csv):
    """Non-label columns of the single row at ts == cut are exempted.

    Real-data root cause (20250701 sse ch5, ofi_60s): the merge consumes
    ticks in file (SeqNo) order and stops at the first tick with
    TransactTime > U; ~1e-4 of ticks carry out-of-order stamps, so
    truncating by SeqNo shifts the stop point and the cut-point snapshot
    absorbs a slightly different tick set in the two runs.  Only the
    boundary row is affected -- rows with ts < cut are bit-exact (the
    causality guarantee).
    """
    text = trunc_csv.read_text().splitlines()
    header = text[0].split(",")
    oir_idx = header.index("oir")
    fields = text[1 + CUT_IDX].split(",")  # ts == CUT_MS -> boundary row
    fields[oir_idx] = "9.999999"
    text[1 + CUT_IDX] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")

    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=900)
    assert diff.identical is True

    # The row right before the boundary is NOT exempt: the same tamper
    # one row earlier must still be caught.
    text = trunc_csv.read_text().splitlines()
    fields = text[1 + CUT_IDX - 1].split(",")
    fields[oir_idx] = "9.999999"
    text[1 + CUT_IDX - 1] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")
    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=900)
    assert diff.identical is False
    assert "oir" in diff.first_diff


def test_compare_prefix_boundary_row_labels_still_compared_in_scope(
    full_csv, trunc_csv
):
    """The boundary exemption covers non-label columns only: when the label
    horizon actually reaches the boundary row (horizons_max_s=0 here) a
    truncated-present label mismatch on that row is still detected."""
    text = trunc_csv.read_text().splitlines()
    header = text[0].split(",")
    lab_idx = header.index("fwd_mid_ret_15s")
    fields = text[1 + CUT_IDX].split(",")  # boundary row
    fields[lab_idx] = "9.876543"
    text[1 + CUT_IDX] = ",".join(fields)
    trunc_csv.write_text("\n".join(text) + "\n")

    diff = compare_prefix(full_csv, trunc_csv, CUT_MS, horizons_max_s=0)
    assert diff.identical is False
    assert "fwd_mid_ret_15s" in diff.first_diff


def test_compare_prefix_missing_row_detected(full_csv, trunc_csv):
    text = trunc_csv.read_text().splitlines()
    del text[6]  # drop row idx 4
    trunc_csv.write_text("\n".join(text) + "\n")
    diff = compare_prefix(full_csv, trunc_csv, CUT_MS)
    assert diff.identical is False
    assert "missing from truncated output" in diff.first_diff


def test_compare_prefix_header_mismatch(full_csv, trunc_csv):
    text = trunc_csv.read_text().splitlines()
    text[0] = text[0].replace("oir", "oir_renamed")
    trunc_csv.write_text("\n".join(text) + "\n")
    diff = compare_prefix(full_csv, trunc_csv, CUT_MS)
    assert diff.identical is False
    assert "header mismatch" in diff.first_diff


# --------------------------------------------------------------------- #
# truncation point selection                                            #
# --------------------------------------------------------------------- #
def test_choose_truncation_points_base(full_csv):
    pts = choose_truncation_points(full_csv, k=4)
    assert [p.label for p in pts] == ["warmup", "mid_am", "post_lunch", "late"]
    tss = [p.ts_ms for p in pts]
    assert tss == sorted(tss)
    assert all(START_MS <= t <= START_MS + (N_SNAP - 1) * 3000 for t in tss)
    assert all(p.tick_seq == -1 for p in pts)  # unresolved sentinel


def test_choose_truncation_points_deterministic_and_fuzz(full_csv):
    a = choose_truncation_points(full_csv, k=7, seed=42)
    b = choose_truncation_points(full_csv, k=7, seed=42)
    assert a == b
    assert len(a) == 7
    labels = [p.label for p in a]
    assert sum(1 for l in labels if l.startswith("fuzz_")) == 3

    c = choose_truncation_points(full_csv, k=7, seed=7)
    assert [p.ts_ms for p in c] != [p.ts_ms for p in a] or True  # fuzz may collide
    assert len(c) == 7

    one = choose_truncation_points(full_csv, k=1)
    assert [p.label for p in one] == ["late"]

    with pytest.raises(ValueError):
        choose_truncation_points(full_csv, k=0)


def test_choose_truncation_points_empty_raises(tmp_path):
    empty = tmp_path / "empty.csv"
    write_interchange_csv(empty, date=DATE, exchange="sse", rows=[], factors=FACTORS)
    with pytest.raises(ValueError):
        choose_truncation_points(empty, k=4)


# --------------------------------------------------------------------- #
# input file truncation                                                 #
# --------------------------------------------------------------------- #
def test_truncate_tick_file_by_seqno(tmp_path):
    src = write_tick_gz(tmp_path / "ticks.csv.gz", n_rows=100)
    dst = tmp_path / "trunc_ticks.csv.gz"
    kept = truncate_tick_file(src, dst, max_seq=40)
    assert kept == 40
    with gzip.open(dst, "rt") as fh:
        lines = fh.read().splitlines()
    assert lines[0].startswith("ExchangeID")
    seqs = [int(l.split(",")[2]) for l in lines[1:]]
    assert seqs == list(range(1, 41))


def test_truncate_snapshot_file_by_time(tmp_path):
    src = write_snapshot_gz(tmp_path / "snaps.csv.gz", n_rows=40,
                            start_ms=START_MS, step_ms=3000)
    dst = tmp_path / "trunc_snaps.csv.gz"
    cut = START_MS + 10 * 3000
    kept = truncate_snapshot_file(src, dst, max_ts_ms=cut)
    assert kept == 11  # rows 0..10 inclusive
    with gzip.open(dst, "rt") as fh:
        lines = fh.read().splitlines()
    assert lines[0].startswith("InstrumentID")
    for l in lines[1:]:
        assert parse_time_ms(l.split(",")[1]) <= cut


def test_parse_time_ms_formats():
    assert parse_time_ms("093000000") == START_MS
    assert parse_time_ms("093015500") == START_MS + 15_500
    assert parse_time_ms("09:30:00") == START_MS
    assert parse_time_ms("09:30:00.250") == START_MS + 250
    assert parse_time_ms("") is None
    # Real dumps drop leading zeros (integer HHMMSSmmm).
    assert parse_time_ms("93000000") == START_MS
    assert parse_time_ms("91400650") == 9 * 3_600_000 + 14 * 60_000 + 650
    assert parse_time_ms("1234567890") is None


# --------------------------------------------------------------------- #
# engine CLI args                                                       #
# --------------------------------------------------------------------- #
def test_engine_cli_args(small_cfg):
    args = engine_cli_args(
        small_cfg,
        exchange="sse", date=DATE, channel=3,
        tick_gz=Path("/t.csv.gz"), snapshot_gz=Path("/s.csv.gz"),
        out_csv=Path("/o.csv"), canaries=True, build_id="abc",
    )
    assert args[0:2] == ["--exchange", "sse"]
    assert "--canaries" in args
    assert args[args.index("--horizons") + 1] == "15,30,60,300,900"
    assert args[args.index("--build-id") + 1] == "abc"
    # cfg.factors == [] => no --factors flag (full default registry)
    assert "--factors" not in args


# --------------------------------------------------------------------- #
# full mask_test_day orchestration with a mocked engine                 #
# --------------------------------------------------------------------- #
def _make_inputs(cfg, date: str):
    day_dir = cfg.data_roots["sse"] / date[:6] / "csv_0603_081500"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot_gz(day_dir / "1_snapshot.csv.gz", n_rows=40, start_ms=START_MS)
    write_tick_gz(day_dir / "1_channel_3.csv.gz", n_rows=600,
                  start_ms=START_MS, step_ms=200)
    return day_dir


def make_fake_engine():
    """A deterministic, prefix-consistent stand-in for hftaf-engine.

    factor fake_f = ts/1e7 (pure function of the snapshot time -> identical
    across full and truncated runs).  With --canaries it also emits
    future_mid_15s = first snapshot time >= ts+15s, which genuinely reads
    the future and therefore MUST break prefix identity on truncation.
    Labels: fwd_*_ret_Hs = first snapshot time >= ts+H, else ABSENT.
    """

    def fake_run_engine(engine_bin, args):
        args = [str(a) for a in args]
        opts = {"canaries": False}
        keymap = {"--ticks": "ticks", "--snapshots": "snaps", "--out": "out"}
        i = 0
        while i < len(args):
            if args[i] in keymap:
                opts[keymap[args[i]]] = args[i + 1]
                i += 2
            elif args[i] == "--canaries":
                opts["canaries"] = True
                i += 1
            else:
                i += 1

        snap_times: list[int] = []
        with gzip.open(opts["snaps"], "rt", encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if len(parts) >= 2:
                    t = parse_time_ms(parts[1])
                    if t is not None:
                        snap_times.append(t)
        snap_times.sort()

        factors = ["fake_f"] + (["future_mid_15s"] if opts["canaries"] else [])
        rows = []
        for idx, ts in enumerate(snap_times):
            row = {
                "instrument": "510300",
                "ts_ms": ts,
                "snap_seq": idx,
                "flags": 0,
                "mid_px": 4.0,
                "last_px": 4.0,
                "bid1_px": 3.999,
                "ask1_px": 4.001,
                "bid1_qty": 1000,
                "ask1_qty": 1000,
                "depth_bid5": 5000,
                "depth_ask5": 5000,
                "fake_f": ts / 1.0e7,
            }

            def first_ge(target: int) -> int | None:
                for t2 in snap_times:
                    if t2 >= target:
                        return t2
                return None

            if opts["canaries"]:
                v = first_ge(ts + 15_000)
                row["future_mid_15s"] = float(v) if v is not None else None
            for h in HORIZONS:
                v = first_ge(ts + h * 1000)
                val = float(v) if v is not None else None
                row[f"fwd_mid_ret_{h}s"] = val
                row[f"fwd_last_ret_{h}s"] = val
            rows.append(row)

        write_interchange_csv(
            Path(opts["out"]), date="20250603", exchange="sse", rows=rows,
            factors=factors,
        )
        return subprocess.CompletedProcess(
            args=[str(engine_bin)], returncode=0, stdout="ok\n", stderr=""
        )

    return fake_run_engine


def test_mask_test_day_passes_with_prefix_consistent_engine(
    small_cfg, monkeypatch
):
    _make_inputs(small_cfg, DATE)
    monkeypatch.setattr(mt, "run_engine", make_fake_engine())

    rep = mt.mask_test_day(small_cfg, DATE, "sse", 3, k=3, include_canaries=True)

    assert rep.passed is True
    assert all(d.identical for d in rep.diffs)
    assert len(rep.points) == 3 and len(rep.diffs) == 3
    assert all(p.tick_seq > 0 for p in rep.points)  # resolved vs tick stream
    assert rep.canary_run is True
    assert rep.canary_failed_as_required is True  # leakage WAS detected
    assert (small_cfg.raw_dir / DATE / "sse_ch3.csv").is_file()


def test_mask_test_day_without_canaries(small_cfg, monkeypatch):
    _make_inputs(small_cfg, DATE)
    monkeypatch.setattr(mt, "run_engine", make_fake_engine())

    rep = mt.mask_test_day(small_cfg, DATE, "sse", 3, k=2, include_canaries=False)
    assert rep.passed is True
    assert rep.canary_run is False
    assert [p.label for p in rep.points] == ["warmup", "late"]


def test_mask_test_day_flags_leaky_engine_without_canary_check(
    small_cfg, monkeypatch
):
    """An engine whose factor depends on the FUTURE must fail the prefix test
    even without the canary column (defence in depth)."""
    _make_inputs(small_cfg, DATE)
    fake = make_fake_engine()

    def leaky(engine_bin, args):
        # sabotage: make fake_f depend on the LAST snapshot of the stream
        cp = fake(engine_bin, args)
        args = [str(a) for a in args]
        out = Path(args[args.index("--out") + 1])
        text = out.read_text().splitlines()
        header = text[0].split(",")
        ts_idx = header.index("ts_ms")
        f_idx = header.index("fake_f")
        last_ts = max(int(l.split(",")[ts_idx]) for l in text[1:])
        new = [text[0]]
        for l in text[1:]:
            fields = l.split(",")
            fields[f_idx] = f"{last_ts / 1e7:.6f}"  # global -> cut-dependent
            new.append(",".join(fields))
        out.write_text("\n".join(new) + "\n")
        return cp

    monkeypatch.setattr(mt, "run_engine", leaky)
    rep = mt.mask_test_day(small_cfg, DATE, "sse", 3, k=3, include_canaries=False)
    assert rep.passed is False
    assert any(not d.identical for d in rep.diffs)


def test_mask_test_day_unknown_job_raises(small_cfg):
    with pytest.raises(FileNotFoundError):
        mt.mask_test_day(small_cfg, "20250101", "sse", 1)


# --------------------------------------------------------------------- #
# run_mask_stage channel filter                                         #
# --------------------------------------------------------------------- #
def _make_inputs_channels(cfg, date: str, channels):
    """Create tick streams for several channels sharing one snapshot file."""
    day_dir = cfg.data_roots["sse"] / date[:6] / "csv_0603_081500"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot_gz(day_dir / "1_snapshot.csv.gz", n_rows=40, start_ms=START_MS)
    for ch in channels:
        write_tick_gz(day_dir / f"1_channel_{ch}.csv.gz", n_rows=600,
                      start_ms=START_MS, step_ms=200)
    return day_dir


def test_run_mask_stage_channels_filter(small_cfg, monkeypatch):
    """channels=[5] restricts the stage to just that (date, channel) job."""
    from hft_autofactor.pipeline import orchestrator

    _make_inputs_channels(small_cfg, DATE, channels=[3, 5])
    monkeypatch.setattr(mt, "run_engine", make_fake_engine())

    report_path = orchestrator.run_mask_stage(
        small_cfg, [DATE], k=2, channels=[5]
    )
    import json

    summary = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["n_jobs"] == 1
    assert summary["n_passed"] == 1
    assert summary["entries"][0]["channel"] == 5
    # channel 5 raw output produced; channel 3 untouched
    assert (small_cfg.raw_dir / DATE / "sse_ch5.csv").is_file()
    assert not (small_cfg.raw_dir / DATE / "sse_ch3.csv").exists()


def test_run_mask_stage_no_filter_runs_all_channels(small_cfg, monkeypatch):
    """channels=None (default) keeps every discovered channel."""
    from hft_autofactor.pipeline import orchestrator

    _make_inputs_channels(small_cfg, DATE, channels=[3, 5])
    monkeypatch.setattr(mt, "run_engine", make_fake_engine())

    report_path = orchestrator.run_mask_stage(small_cfg, [DATE], k=2)
    import json

    summary = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["n_jobs"] == 2
    assert summary["n_passed"] == 2
    assert sorted(e["channel"] for e in summary["entries"]) == [3, 5]
