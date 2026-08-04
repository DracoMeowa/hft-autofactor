"""hftaf factors --cache build/use orchestration tests.

The engine is MOCKED: build mode writes a cache meta.json recording the
input sizes, use mode writes the requested output file.  This proves the
stage wiring -- argument threading, skip-if-built, and replay output
placement -- without needing the C++ binary.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conftest import write_snapshot_gz, write_tick_gz

from hft_autofactor.pipeline import orchestrator
from hft_autofactor.validation.mask_test import engine_cli_args

DATE = "20250603"


# --------------------------------------------------------------------- #
# fixtures                                                              #
# --------------------------------------------------------------------- #
def _make_inputs(cfg, date: str = DATE, channel: int = 3):
    day_dir = cfg.data_roots["sse"] / date[:6] / "csv_0603_081500"
    day_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot_gz(day_dir / "1_snapshot.csv.gz", n_rows=40)
    write_tick_gz(day_dir / f"1_channel_{channel}.csv.gz", n_rows=600)
    return day_dir


def make_fake_engine():
    """Mock engine: build -> meta.json with real input sizes; use -> out file."""

    def fake_run_engine(engine_bin, args):
        args = [str(a) for a in args]

        def val(flag):
            return args[args.index(flag) + 1] if flag in args else None

        build_dir, use_dir, out = val("--build-cache"), val("--use-cache"), val("--out")
        if build_dir is not None:
            assert "--out" not in args
            tick_bytes = Path(val("--ticks")).stat().st_size
            snap_bytes = Path(val("--snapshots")).stat().st_size
            d = Path(build_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "meta.json").write_text(
                json.dumps(
                    {
                        "kind": "hftaf-cache",
                        "tick_bytes": tick_bytes,
                        "snapshot_bytes": snap_bytes,
                    }
                ),
                encoding="utf-8",
            )
        elif use_dir is not None:
            assert "--ticks" not in args and "--snapshots" not in args
            assert Path(use_dir, "meta.json").is_file(), "replay without a cache"
            assert out is not None
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("date,exchange,instrument,ts_ms\n", encoding="utf-8")
        else:
            assert out is not None
            p = Path(out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("raw\n", encoding="utf-8")
        return subprocess.CompletedProcess([str(engine_bin)], 0, "ok\n", "")

    return fake_run_engine


# --------------------------------------------------------------------- #
# path helpers                                                          #
# --------------------------------------------------------------------- #
def test_cache_dir_for_layout(small_cfg):
    _make_inputs(small_cfg)
    job = orchestrator.discover_jobs(small_cfg, [DATE])[0]
    assert orchestrator.cache_dir_for(small_cfg, job) == (
        small_cfg.out_root / "cache" / DATE / "sse_ch3"
    )
    assert orchestrator.cache_dir_for(small_cfg, job, ["588000"]) == (
        small_cfg.out_root / "cache" / DATE / "sse_ch3" / "588000"
    )
    # deterministic under code order
    scoped = orchestrator.cache_dir_for(small_cfg, job, ["588010", "588000"])
    assert scoped.name == "588000+588010"


def test_replay_out_csv_placement(small_cfg):
    _make_inputs(small_cfg)
    job = orchestrator.discover_jobs(small_cfg, [DATE])[0]
    # whole-channel replay targets the standard production raw CSV
    assert orchestrator.replay_out_csv(small_cfg, job) == job.out_csv
    # instrument-scoped replay must NOT clobber the full production CSV
    scoped = orchestrator.replay_out_csv(small_cfg, job, ["588000"])
    assert scoped.name == "sse_ch3_replay_588000.csv"
    assert scoped.parent == job.out_csv.parent


# --------------------------------------------------------------------- #
# cache build mode                                                      #
# --------------------------------------------------------------------- #
def test_factors_cache_build_then_skip_if_built(small_cfg, monkeypatch):
    _make_inputs(small_cfg)
    monkeypatch.setattr(orchestrator, "run_engine", make_fake_engine())

    res = orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="build")
    assert [r.status for r in res] == ["ok"]
    meta = small_cfg.out_root / "cache" / DATE / "sse_ch3" / "meta.json"
    assert meta.is_file()
    # build mode produces no factor CSV
    job = orchestrator.discover_jobs(small_cfg, [DATE])[0]
    assert not job.out_csv.exists()

    # second run: skip-if-built against the cache meta
    res2 = orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="build")
    assert [r.status for r in res2] == ["skipped"]

    # --overwrite rebuilds
    res3 = orchestrator.run_factor_stage(
        small_cfg, [DATE], channels=[3], cache="build", overwrite=True
    )
    assert [r.status for r in res3] == ["ok"]


def test_factors_cache_build_scoped_is_separate_cache(small_cfg, monkeypatch):
    _make_inputs(small_cfg)
    seen: list[list[str]] = []
    fake = make_fake_engine()

    def spy(engine_bin, args):
        seen.append([str(a) for a in args])
        return fake(engine_bin, args)

    monkeypatch.setattr(orchestrator, "run_engine", spy)

    res = orchestrator.run_factor_stage(
        small_cfg, [DATE], channels=[3], cache="build", cache_instruments=["588000"]
    )
    assert [r.status for r in res] == ["ok"]
    args = seen[0]
    assert "--cache-instruments" in args
    assert args[args.index("--cache-instruments") + 1] == "588000"
    scoped_meta = (
        small_cfg.out_root / "cache" / DATE / "sse_ch3" / "588000" / "meta.json"
    )
    assert scoped_meta.is_file()

    # scoped rebuild skips against its own meta ...
    res2 = orchestrator.run_factor_stage(
        small_cfg, [DATE], channels=[3], cache="build", cache_instruments=["588000"]
    )
    assert [r.status for r in res2] == ["skipped"]
    # ... and does NOT count as having built the whole-channel cache
    res3 = orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="build")
    assert [r.status for r in res3] == ["ok"]


# --------------------------------------------------------------------- #
# cache use (replay) mode                                               #
# --------------------------------------------------------------------- #
def test_factors_cache_use_whole_channel_writes_standard_raw(small_cfg, monkeypatch):
    _make_inputs(small_cfg)
    monkeypatch.setattr(orchestrator, "run_engine", make_fake_engine())
    orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="build")

    res = orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="use")
    assert [r.status for r in res] == ["ok"]
    job = orchestrator.discover_jobs(small_cfg, [DATE])[0]
    assert job.out_csv.is_file()  # standard production path

    # replay is NEVER skipped: recomputation is the point (engine rebuilds)
    res2 = orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="use")
    assert [r.status for r in res2] == ["ok"]


def test_factors_cache_use_scoped_writes_side_file(small_cfg, monkeypatch):
    _make_inputs(small_cfg)
    monkeypatch.setattr(orchestrator, "run_engine", make_fake_engine())
    orchestrator.run_factor_stage(
        small_cfg, [DATE], channels=[3], cache="build", cache_instruments=["588000"]
    )

    res = orchestrator.run_factor_stage(
        small_cfg, [DATE], channels=[3], cache="use", cache_instruments=["588000"]
    )
    assert [r.status for r in res] == ["ok"]
    job = orchestrator.discover_jobs(small_cfg, [DATE])[0]
    side = job.out_csv.parent / "sse_ch3_replay_588000.csv"
    assert side.is_file()
    assert not job.out_csv.exists()  # production CSV untouched by scoped replay


def test_factors_cache_argument_threading(small_cfg, monkeypatch):
    """build: --ticks/--snapshots present, --out omitted; use: the reverse."""
    _make_inputs(small_cfg)
    seen: list[list[str]] = []
    fake = make_fake_engine()

    def spy(engine_bin, args):
        seen.append([str(a) for a in args])
        return fake(engine_bin, args)

    monkeypatch.setattr(orchestrator, "run_engine", spy)

    orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="build")
    orchestrator.run_factor_stage(small_cfg, [DATE], channels=[3], cache="use")

    build_args, use_args = seen[0], seen[1]
    assert "--build-cache" in build_args
    assert "--ticks" in build_args and "--snapshots" in build_args
    assert "--out" not in build_args

    assert "--use-cache" in use_args
    assert use_args[use_args.index("--use-cache") + 1] == build_args[
        build_args.index("--build-cache") + 1
    ]
    assert "--ticks" not in use_args and "--snapshots" not in use_args
    assert "--out" in use_args


# --------------------------------------------------------------------- #
# engine_cli_args cache plumbing                                        #
# --------------------------------------------------------------------- #
def test_engine_cli_args_cache_modes(small_cfg):
    build = engine_cli_args(
        small_cfg, exchange="sse", date=DATE, channel=3,
        tick_gz=Path("/t.csv.gz"), snapshot_gz=Path("/s.csv.gz"),
        out_csv=Path("/o.csv"), build_cache_dir=Path("/cache/d"),
        cache_instruments=["588000"],
    )
    assert "--build-cache" in build and "--out" not in build
    assert build[build.index("--cache-instruments") + 1] == "588000"

    use = engine_cli_args(
        small_cfg, exchange="sse", date=DATE, channel=3,
        tick_gz=Path("/t.csv.gz"), snapshot_gz=Path("/s.csv.gz"),
        out_csv=Path("/o.csv"), use_cache_dir=Path("/cache/d"),
    )
    assert "--use-cache" in use
    assert "--ticks" not in use and "--snapshots" not in use
    assert "--out" in use  # replay still writes an output CSV

    with pytest.raises(ValueError):
        engine_cli_args(
            small_cfg, exchange="sse", date=DATE, channel=3,
            tick_gz=Path("/t.csv.gz"), snapshot_gz=Path("/s.csv.gz"),
            out_csv=Path("/o.csv"),
            build_cache_dir=Path("/c"), use_cache_dir=Path("/c"),
        )


# --------------------------------------------------------------------- #
# CLI flag parsing                                                      #
# --------------------------------------------------------------------- #
def test_cli_factors_cache_flags_parse():
    from hft_autofactor.pipeline.cli import build_parser

    p = build_parser()
    ns = p.parse_args(
        ["factors", "--dates", DATE, "--cache", "build",
         "--cache-instruments", "588000,588010"]
    )
    assert ns.cache == "build"
    assert ns.cache_instruments == "588000,588010"

    ns2 = p.parse_args(["factors", "--dates", DATE, "--cache", "use"])
    assert ns2.cache == "use"
    assert ns2.cache_instruments is None
