"""Explore-lane pre-screen and hftaf-explore CLI tests.

Synthetic partitions are written directly under ``cfg.parquet_path`` with a
planted signal: ``last_px`` carries a fresh N(0,1) signal each row and every
label equals signal + small noise, so an identity prototype on ``last_px``
has RankIC ~ 1 at every horizon while library factors (pure noise) have
near-zero correlation with it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_autofactor.config import PipelineConfig
from hft_autofactor.eval.gating import TrialLedger
from hft_autofactor.explore.cli import main as explore_main
from hft_autofactor.explore.layout import panel_path, spec_meta_path, spec_path
from hft_autofactor.explore.registry import explore_prototype
from hft_autofactor.explore.runner import run_prototype
from hft_autofactor.explore.screen import (
    ScreenConfig,
    library_correlations,
    screen_prototype,
)

DATES = [f"202506{d:02d}" for d in range(2, 14)]  # 12 synthetic days
INSTRUMENTS = ("510300", "510050", "159915", "588000")
N_ROWS = 80
START_MS = 34_200_000
HORIZONS = (15, 30, 60, 300, 900)


# --------------------------------------------------------------------- #
# synthetic partition builder                                           #
# --------------------------------------------------------------------- #
def write_synthetic_partitions(
    cfg: PipelineConfig, dates=DATES, *, seed: int = 5, label_noise: float = 0.25
) -> None:
    rng = np.random.default_rng(seed)
    for date in dates:
        records = []
        for inst in INSTRUMENTS:
            mid = 4.0
            for r in range(N_ROWS):
                signal = rng.standard_normal()
                mid += rng.normal(0.0, 0.0005)
                ts = START_MS + r * 3000
                rec = {
                    "date": date,
                    "exchange": "sse",
                    "instrument": inst,
                    "ts_ms": ts,
                    "snap_seq": 1000 + r,
                    "flags": 0,
                    "mid_px": mid,
                    "last_px": signal,  # planted signal column
                    "bid1_px": mid - 0.001,
                    "ask1_px": mid + 0.001,
                    "bid1_qty": 10000,
                    "ask1_qty": 8000,
                    "depth_bid5": 50000,
                    "depth_ask5": 42000,
                    "oir": rng.standard_normal(),
                    "wdi": rng.standard_normal(),
                    "quoted_spread_ticks": 2.0,
                    "channel": 1,
                }
                for h in HORIZONS:
                    rec[f"fwd_mid_ret_{h}s"] = signal + label_noise * rng.standard_normal()
                    rec[f"fwd_last_ret_{h}s"] = signal + label_noise * rng.standard_normal()
                records.append(rec)
        path = cfg.parquet_path(date)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(path)


def signal_proto():
    return explore_prototype(
        name="planted_signal_f",
        mechanism="reads the planted signal column (test double)",
        info_set="last_px",
        inspiration="synthetic fixture",
        compute=lambda part: part["last_px"],
    )


def duplicate_proto():
    return explore_prototype(
        name="oir_copy_f",
        mechanism="exact copy of the library oir (must be flagged duplicate)",
        info_set="oir",
        inspiration="synthetic fixture",
        compute=lambda part: part["oir"],
    )


def weak_proto():
    return explore_prototype(
        name="mid_drift_f",
        mechanism="signed mid drift (uncorrelated with the planted labels)",
        info_set="mid_px",
        inspiration="synthetic fixture",
        compute=lambda part: part["mid_px"].log().diff(1),
    )


@pytest.fixture
def explore_cfg(small_cfg):
    write_synthetic_partitions(small_cfg)
    return small_cfg


def _run(cfg, proto):
    result = run_prototype(cfg, proto, DATES, chunk_days=4, k=3)
    assert result.status == "ok", result.message
    return result


# --------------------------------------------------------------------- #
# library correlations                                                  #
# --------------------------------------------------------------------- #
def test_library_correlations_detect_identity_duplicate(explore_cfg):
    _run(explore_cfg, duplicate_proto())
    from hft_autofactor.explore.runner import load_explore_panel

    panel = load_explore_panel(explore_cfg, "oir_copy_f", DATES)
    corrs = library_correlations(panel, "oir_copy_f", max_obs=50_000)
    assert corrs["oir"] == pytest.approx(1.0, abs=1e-12)
    assert abs(corrs["wdi"]) < 0.1
    assert "planted_signal_f" not in corrs  # only DEFAULT_FACTORS are scored


def test_library_correlations_subsample_is_deterministic(explore_cfg):
    _run(explore_cfg, signal_proto())
    from hft_autofactor.explore.runner import load_explore_panel

    panel = load_explore_panel(explore_cfg, "planted_signal_f", DATES)
    a = library_correlations(panel, "planted_signal_f", max_obs=500)
    b = library_correlations(panel, "planted_signal_f", max_obs=500)
    assert set(a) == set(b)
    for f in a:
        if math.isfinite(a[f]):
            assert a[f] == b[f]
        else:
            assert not math.isfinite(b[f])


# --------------------------------------------------------------------- #
# screen_prototype                                                      #
# --------------------------------------------------------------------- #
def test_screen_passes_strong_novel_signal(explore_cfg):
    _run(explore_cfg, signal_proto())
    report = screen_prototype(explore_cfg, signal_proto(), DATES)

    assert report.status == "ok"
    assert report.passed is True
    assert report.duplicate_check["duplicated"] is False
    assert report.duplicate_check["max_abs_corr"] < 0.85
    assert len(report.horizons) == len(HORIZONS)
    for row in report.horizons:
        assert row["is_mean_ic"] > 0.8
        assert row["oos_mean_ic"] > 0.4
        assert row["retention_ok"] is True
        assert row["passed"] is True
    assert report.report_path is not None and report.report_path.is_file()
    payload = json.loads(report.report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert report.report_path.with_suffix(".csv").is_file()


def test_screen_rejects_library_duplicate(explore_cfg):
    _run(explore_cfg, duplicate_proto())
    report = screen_prototype(explore_cfg, duplicate_proto(), DATES)

    assert report.status == "rejected_duplicate"
    assert report.passed is False
    assert report.duplicate_check["duplicated"] is True
    assert report.duplicate_check["library_factor"] == "oir"
    assert report.duplicate_check["max_abs_corr"] > 0.85
    assert any("duplicate" in r for r in report.reasons)


def test_screen_fails_weak_signal_without_duplicate_flag(explore_cfg):
    _run(explore_cfg, weak_proto())
    report = screen_prototype(explore_cfg, weak_proto(), DATES)

    assert report.status == "ok"  # not a duplicate...
    assert report.passed is False  # ...but no horizon survives IS/OOS
    assert report.duplicate_check["duplicated"] is False
    assert any("no horizon passed" in r for r in report.reasons)
    assert all(row["passed"] is False for row in report.horizons)


def test_screen_split_is_purged_by_at_least_one_day(explore_cfg):
    _run(explore_cfg, signal_proto())
    report = screen_prototype(explore_cfg, signal_proto(), DATES)

    split = report.split
    assert split["embargo_days"] >= 1
    train, test = split["train_dates"], split["test_dates"]
    assert train and test
    # >= 1 calendar day embargoed strictly between train end and test start
    assert DATES.index(test[0]) - DATES.index(train[-1]) >= 2
    embargoed = set(DATES[DATES.index(train[-1]) + 1 : DATES.index(test[0])])
    assert embargoed and not (embargoed & set(train)) and not (embargoed & set(test))


def test_screen_insufficient_dates(explore_cfg):
    two = DATES[:2]
    _run(explore_cfg, signal_proto())  # partitions for all dates exist
    report = screen_prototype(explore_cfg, signal_proto(), two)
    assert report.status == "insufficient_data"
    assert report.passed is False
    assert any("no purged IS/OOS split" in r for r in report.reasons)


def test_screen_logs_trials_before_thresholds(explore_cfg):
    _run(explore_cfg, signal_proto())
    screen_prototype(explore_cfg, signal_proto(), DATES)
    ledger = TrialLedger(explore_cfg.reports_dir / "trial_ledger.jsonl")
    assert ledger.n_trials("explore_screen") == len(HORIZONS)


def test_screen_missing_partitions_raises(explore_cfg):
    proto = explore_prototype(
        name="never_ran_f", mechanism="m", info_set="mid_px",
        inspiration="i", compute=lambda part: part["mid_px"],
    )
    with pytest.raises(FileNotFoundError, match="hftaf-explore run"):
        screen_prototype(explore_cfg, proto, DATES)


def test_screen_respects_custom_thresholds(explore_cfg):
    """An absurdly tight retention gate flips the verdict to FAIL."""
    _run(explore_cfg, signal_proto())
    tight = ScreenConfig(min_retention=1.5)  # impossible retention
    report = screen_prototype(explore_cfg, signal_proto(), DATES, screen_cfg=tight)
    assert report.passed is False


# --------------------------------------------------------------------- #
# hftaf-explore CLI                                                     #
# --------------------------------------------------------------------- #
@pytest.fixture
def cli_env(explore_cfg, tmp_path):
    """Config YAML + env for driving explore_main() end to end."""
    cfg = explore_cfg
    cfg_yaml = tmp_path / "pipeline-explore.yaml"
    cfg_yaml.write_text(
        "data_roots:\n"
        f"  sse: {cfg.data_roots['sse']}\n"
        f"  szse: {cfg.data_roots['szse']}\n"
        f"out_root: {cfg.out_root}\n"
        f"engine_bin: {cfg.engine_bin}\n"
        "horizons: [15, 30, 60, 300, 900]\n"
        "max_workers: 1\n",
        encoding="utf-8",
    )
    return cfg, cfg_yaml


def _spec_file(tmp_path: Path, name: str = "cli_signal_f", inspiration: str = "cli test") -> Path:
    spec = tmp_path / f"{name}.py"
    spec.write_text(
        "import polars as pl\n"
        "from hft_autofactor.explore.registry import explore_prototype\n\n"
        "PROTOTYPE = explore_prototype(\n"
        f"    name={name!r},\n"
        "    mechanism='planted-signal reader for CLI tests',\n"
        "    info_set='last_px',\n"
        f"    inspiration={inspiration!r},\n"
        "    compute=lambda part: part['last_px'],\n"
        ")\n",
        encoding="utf-8",
    )
    return spec


def test_cli_list_shows_builtins(cli_env, capsys):
    cfg, cfg_yaml = cli_env
    rc = explore_main(["--config", str(cfg_yaml), "list"])
    assert rc == 0
    out = capsys.readouterr().out
    for name in ("log_mid_ret_60s", "spread_z_300s", "depth_imbalance_5l"):
        assert name in out
    assert "prototype(s)" in out


def test_cli_add_persists_spec_and_sidecar(cli_env):
    cfg, cfg_yaml = cli_env
    spec = _spec_file(cfg.out_root.parent)  # tmp_path
    rc = explore_main(["--config", str(cfg_yaml), "add", "--spec", str(spec)])
    assert rc == 0
    assert spec_path(cfg, "cli_signal_f").is_file()
    meta = json.loads(spec_meta_path(cfg, "cli_signal_f").read_text(encoding="utf-8"))
    assert meta["name"] == "cli_signal_f"
    assert meta["mechanism"] and meta["info_set"] and meta["inspiration"]

    # duplicate add refused without --overwrite
    rc2 = explore_main(["--config", str(cfg_yaml), "add", "--spec", str(spec)])
    assert rc2 == 1
    rc3 = explore_main(
        ["--config", str(cfg_yaml), "add", "--spec", str(spec), "--overwrite"]
    )
    assert rc3 == 0


def test_cli_add_refuses_incomplete_spec(cli_env, capsys):
    cfg, cfg_yaml = cli_env
    bad = cfg.out_root.parent / "incomplete.py"
    bad.write_text(
        "from hft_autofactor.explore.registry import explore_prototype\n"
        "PROTOTYPE = dict(name='incomplete_f', mechanism='m', info_set='mid_px',\n"
        "                 inspiration='', compute=lambda part: part['mid_px'])\n",
        encoding="utf-8",
    )
    rc = explore_main(["--config", str(cfg_yaml), "add", "--spec", str(bad)])
    assert rc == 1
    assert not spec_path(cfg, "incomplete_f").exists()
    err = capsys.readouterr().err
    assert "refused" in err and "inspiration" in err


def test_cli_run_and_screen_end_to_end(cli_env, capsys):
    cfg, cfg_yaml = cli_env
    spec = _spec_file(cfg.out_root.parent)
    assert explore_main(["--config", str(cfg_yaml), "add", "--spec", str(spec)]) == 0

    dates_spec = f"{DATES[0]}..{DATES[-1]}"
    capsys.readouterr()
    rc = explore_main(
        ["--config", str(cfg_yaml), "run", "--dates", dates_spec,
         "--protos", "cli_signal_f", "--chunk-days", "5"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out and "cli_signal_f" in out
    assert panel_path(cfg, "cli_signal_f", DATES[0]).is_file()

    capsys.readouterr()
    rc2 = explore_main(
        ["--config", str(cfg_yaml), "screen", "--dates", dates_spec,
         "--protos", "cli_signal_f"]
    )
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "PASS" in out2 and "cli_signal_f" in out2


def test_cli_run_rejects_leaky_prototype(cli_env, capsys):
    cfg, cfg_yaml = cli_env
    leaky = cfg.out_root.parent / "leaky_cli_f.py"
    leaky.write_text(
        "from hft_autofactor.explore.registry import explore_prototype\n"
        "PROTOTYPE = explore_prototype(\n"
        "    name='leaky_cli_f',\n"
        "    mechanism='canary: forward shift',\n"
        "    info_set='mid_px',\n"
        "    inspiration='mask-test canary analogue',\n"
        "    compute=lambda part: part['mid_px'].shift(-5),\n"
        ")\n",
        encoding="utf-8",
    )
    assert explore_main(["--config", str(cfg_yaml), "add", "--spec", str(leaky)]) == 0

    dates_spec = f"{DATES[0]}..{DATES[-1]}"
    capsys.readouterr()
    rc = explore_main(
        ["--config", str(cfg_yaml), "run", "--dates", dates_spec,
         "--protos", "leaky_cli_f"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "REJECTED" in out and "causality" in out
    assert not panel_path(cfg, "leaky_cli_f", DATES[0]).exists()


def test_cli_run_unknown_prototype_is_usage_error(cli_env, capsys):
    cfg, cfg_yaml = cli_env
    rc = explore_main(
        ["--config", str(cfg_yaml), "run", "--dates", DATES[0],
         "--protos", "does_not_exist"]
    )
    assert rc == 2
    assert "unknown prototype" in capsys.readouterr().err


def test_cli_screen_missing_partitions_errors(cli_env, capsys):
    cfg, cfg_yaml = cli_env
    spec = _spec_file(cfg.out_root.parent, name="unran_f")
    assert explore_main(["--config", str(cfg_yaml), "add", "--spec", str(spec)]) == 0
    capsys.readouterr()
    rc = explore_main(
        ["--config", str(cfg_yaml), "screen", "--dates", f"{DATES[0]}..{DATES[-1]}",
         "--protos", "unran_f"]
    )
    assert rc == 1
    assert "ERROR" in capsys.readouterr().out


def test_cli_bad_dates_is_usage_error(cli_env, capsys):
    cfg, cfg_yaml = cli_env
    rc = explore_main(
        ["--config", str(cfg_yaml), "run", "--dates", "notadate"]
    )
    assert rc == 2
    assert "error" in capsys.readouterr().err
