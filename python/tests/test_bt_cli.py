"""CLI tests for ``hftaf-backtest``, focused on the ``--panel-dir`` route (#130).

``--panel-dir`` loads the panel from an archived directory of day partitions
(``dt=YYYYMMDD.parquet`` flat files or ``dt=YYYYMMDD/`` dirs) instead of the
pipeline parquet store -- the route used to retro-test explore-lane prototype
factors on the exact factor values that were screened, without re-running the
factor engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from hft_autofactor.backtest.cli import _resolve_dates, main

# --------------------------------------------------------------------------- #
# _resolve_dates                                                              #
# --------------------------------------------------------------------------- #
def test_resolve_dates_flat_parquet_files(tmp_path: Path) -> None:
    d = tmp_path / "panels"
    d.mkdir()
    for date in ("20250701", "20250702", "20250703", "20250905"):
        (d / ("dt=%s.parquet" % date)).write_bytes(b"")
    (d / "notes.txt").write_text("not a partition", encoding="utf-8")

    got = _resolve_dates(d, "20250701..20250703")
    assert got == ["20250701", "20250702", "20250703"]
    # range bounds are inclusive; out-of-range partitions excluded
    assert _resolve_dates(d, "20250702..20250901") == ["20250702", "20250703"]


def test_resolve_dates_dir_partitions(tmp_path: Path) -> None:
    d = tmp_path / "parquet"
    for date in ("20250701", "20250702"):
        (d / ("dt=%s" % date)).mkdir(parents=True)
    assert _resolve_dates(d, "20250701..20250731") == ["20250701", "20250702"]


def test_resolve_dates_empty_range_raises(tmp_path: Path) -> None:
    d = tmp_path / "panels"
    d.mkdir()
    (d / "dt=20250701.parquet").write_bytes(b"")
    with pytest.raises(ValueError):
        _resolve_dates(d, "20260101..20260131")


def test_resolve_dates_explicit_list_passthrough(tmp_path: Path) -> None:
    # explicit lists are not expanded against the directory
    assert _resolve_dates(tmp_path, "20250701, 20250702") == [
        "20250701",
        "20250702",
    ]


# --------------------------------------------------------------------------- #
# end-to-end: --panel-dir + --matrix                                          #
# --------------------------------------------------------------------------- #
PIPELINE_YAML = """
data_roots:
  sse: {root}/data/sse
  szse: {root}/data/szse
out_root: {root}/out
engine_bin: {root}/bin/hftaf-engine
horizons: [15, 30, 60, 300, 900]
commission_scenarios: [institutional]
max_workers: 1
"""

PARAMS_YAML = """
fees:
  stamp_duty: {rate_per_side: 0.0}
  handling_fee:
    rate_per_side: 0.00004
    exempt_categories: [money_etf, bond_etf]
  regulatory_fee:
    rate_per_side_base: 0.0
    rate_per_side_conservative: 0.00002
  transfer_fee:
    rate_per_side_base: 0.0
    rate_per_side_conservative: 0.00001
  commission:
    scenarios:
      retail_default: {rate_per_side: 0.00025, min_per_order_cny: 5.0}
      retail_negotiated: {rate_per_side: 0.00010, min_per_order_cny: 5.0}
      institutional: {rate_per_side: 0.00005, min_per_order_cny: 0.0}
settlement:
  t_plus_1: [equity_etf]
  t_plus_0: [bond_etf, money_etf, gold_etf, commodity_etf,
             commodity_futures_etf, cross_border_etf]
securities_lending:
  borrow_rate_annual: 0.08
  min_charge_days: 1.0
  day_count_base: 360.0
  source: 'test'
"""


def _day_panel(date: str, n: int = 40) -> pl.DataFrame:
    """Minimal interchange-schema day panel with a prototype factor column."""
    ts = [34_200_000 + 3000 * i for i in range(n)]
    return pl.DataFrame(
        {
            "date": [date] * n,
            "exchange": ["sse"] * n,
            "instrument": ["588000"] * n,
            "ts_ms": ts,
            "snap_seq": list(range(n)),
            "flags": [0] * n,
            "mid_px": [1.300] * n,
            "last_px": [1.300] * n,
            "bid1_px": [1.299] * n,
            "ask1_px": [1.301] * n,
            "bid1_qty": [10_000.0] * n,
            "ask1_qty": [10_000.0] * n,
            "depth_bid5": [1e6] * n,
            "depth_ask5": [1e6] * n,
            "proto_sig": [float(i % 7) - 3.0 for i in range(n)],
            "fwd_mid_ret_15s": [1.0e-4] * n,
        }
    )


def _write_panel_dir(panel_dir: Path, dates=("20250701", "20250702")) -> None:
    panel_dir.mkdir(parents=True, exist_ok=True)
    for d in dates:
        _day_panel(d).write_parquet(panel_dir / ("dt=%s.parquet" % d))
    (panel_dir / "prototypes").mkdir(exist_ok=True)  # stray non-partition entry


def _write_env(tmp_path: Path) -> tuple[Path, Path]:
    cfg_yaml = tmp_path / "pipeline.yaml"
    cfg_yaml.write_text(PIPELINE_YAML.format(root=tmp_path), encoding="utf-8")
    params_yaml = tmp_path / "params.yaml"
    params_yaml.write_text(PARAMS_YAML, encoding="utf-8")
    return cfg_yaml, params_yaml


def test_panel_dir_matrix_end_to_end(tmp_path: Path) -> None:
    """Matrix mode reads archived flat partitions, ledger gets honest-N
    entries even when the tiny synthetic panel yields zero trades."""
    panel_dir = tmp_path / "panels" / "proto_sig"
    _write_panel_dir(panel_dir)
    cfg_yaml, params_yaml = _write_env(tmp_path)
    out_dir = tmp_path / "matrix_out"

    rc = main(
        [
            "--config", str(cfg_yaml),
            "--factor", "proto_sig",
            "--horizon", "15",
            "--dates", "20250701..20250702",
            "--matrix",
            "--direction", "1",
            "--panel-dir", str(panel_dir),
            "--params-yaml", str(params_yaml),
            "--matrix-out", str(out_dir),
        ]
    )
    assert rc == 0

    report = json.loads((out_dir / "matrix.json").read_text(encoding="utf-8"))
    assert report["factor"] == "proto_sig"
    assert report["horizon_s"] == 15
    assert report["ic_direction"] == 1
    assert report["primary_cell"] == {
        "direction": "long",
        "tau": 0.01,
        "regime": "all",
    }
    # tiny panel (40 rows/day < intraday warm-up) -> no trades anywhere:
    # cells table is empty but the run is still reported and ledgered
    assert report["cells"] == []
    assert report["primary_gate"]["passed"] is False
    assert (out_dir / "matrix.md").is_file()

    # honest-N: the zero-trade run is still ledgered
    ledger_path = tmp_path / "out" / "reports" / "trial_ledger.jsonl"
    assert ledger_path.is_file()
    lines = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1  # one summary entry: no trades in any cell
    assert lines[0]["stage"] == "matrix_cell"
    assert lines[0]["metrics"]["n_trades"] == 0


def test_panel_dir_matrix_eval_dates_resolve_against_panel_dir(
    tmp_path: Path,
) -> None:
    panel_dir = tmp_path / "panels" / "proto_sig"
    _write_panel_dir(panel_dir)
    cfg_yaml, params_yaml = _write_env(tmp_path)

    rc = main(
        [
            "--config", str(cfg_yaml),
            "--factor", "proto_sig",
            "--horizon", "15",
            "--dates", "20250701..20250702",
            "--eval-dates", "20250702..20250702",
            "--matrix",
            "--direction", "-1",
            "--panel-dir", str(panel_dir),
            "--params-yaml", str(params_yaml),
            "--matrix-out", str(tmp_path / "mx"),
        ]
    )
    assert rc == 0
    report = json.loads((tmp_path / "mx" / "matrix.json").read_text(encoding="utf-8"))
    # reversal direction fixes the primary cell to short
    assert report["primary_cell"]["direction"] == "short"


def test_panel_dir_missing_factor_column_errors(tmp_path: Path) -> None:
    panel_dir = tmp_path / "panels" / "other"
    _write_panel_dir(panel_dir)
    cfg_yaml, params_yaml = _write_env(tmp_path)

    rc = main(
        [
            "--config", str(cfg_yaml),
            "--factor", "not_in_panel",
            "--horizon", "15",
            "--dates", "20250701..20250702",
            "--matrix",
            "--panel-dir", str(panel_dir),
            "--params-yaml", str(params_yaml),
        ]
    )
    assert rc == 2


def test_panel_dir_not_a_directory_errors(tmp_path: Path) -> None:
    cfg_yaml, params_yaml = _write_env(tmp_path)
    rc = main(
        [
            "--config", str(cfg_yaml),
            "--factor", "proto_sig",
            "--horizon", "15",
            "--dates", "20250701",
            "--matrix",
            "--panel-dir", str(tmp_path / "nope"),
            "--params-yaml", str(params_yaml),
        ]
    )
    assert rc == 2


def test_panel_dir_missing_partition_errors(tmp_path: Path) -> None:
    panel_dir = tmp_path / "panels" / "proto_sig"
    _write_panel_dir(panel_dir, dates=("20250701",))
    cfg_yaml, params_yaml = _write_env(tmp_path)

    rc = main(
        [
            "--config", str(cfg_yaml),
            "--factor", "proto_sig",
            "--horizon", "15",
            "--dates", "20250701,20250705",  # 20250705 has no partition
            "--matrix",
            "--panel-dir", str(panel_dir),
            "--params-yaml", str(params_yaml),
        ]
    )
    assert rc == 2
