"""Output layout for the explore lane.

Everything the explore lane writes lives under ``{out_root}/explore``::

    {out_root}/explore/
      prototypes/{name}.py        # persisted compute specs (added via CLI)
      prototypes/{name}.json      # metadata sidecar (written on add)
      panels/{name}/dt={date}.parquet  # augmented day partitions
      reports/run_{name}_{stamp}.json
      reports/screen_{name}_{stamp}.json (+ .csv)

Nothing is ever written into the read-only exchange data roots.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..config import PipelineConfig

__all__ = [
    "explore_root",
    "prototypes_dir",
    "spec_path",
    "spec_meta_path",
    "panels_dir",
    "panel_path",
    "reports_dir",
    "run_report_path",
    "screen_report_path",
    "sanitize_for_json",
]


def explore_root(cfg: PipelineConfig) -> Path:
    return Path(cfg.out_root) / "explore"


def prototypes_dir(cfg: PipelineConfig) -> Path:
    return explore_root(cfg) / "prototypes"


def spec_path(cfg: PipelineConfig, name: str) -> Path:
    return prototypes_dir(cfg) / f"{name}.py"


def spec_meta_path(cfg: PipelineConfig, name: str) -> Path:
    return prototypes_dir(cfg) / f"{name}.json"


def panels_dir(cfg: PipelineConfig, name: str) -> Path:
    return explore_root(cfg) / "panels" / name


def panel_path(cfg: PipelineConfig, name: str, date: str) -> Path:
    return panels_dir(cfg, name) / f"dt={date}.parquet"


def reports_dir(cfg: PipelineConfig) -> Path:
    return explore_root(cfg) / "reports"


def run_report_path(cfg: PipelineConfig, name: str, dates: list[str]) -> Path:
    stamp = _date_stamp(dates)
    return reports_dir(cfg) / f"run_{name}_{stamp}.json"


def screen_report_path(cfg: PipelineConfig, name: str, dates: list[str]) -> Path:
    stamp = _date_stamp(dates)
    return reports_dir(cfg) / f"screen_{name}_{stamp}.json"


def _date_stamp(dates: list[str]) -> str:
    if not dates:
        return "nodates"
    return f"{dates[0]}_{dates[-1]}_{time.strftime('%H%M%S')}"


def sanitize_for_json(obj):
    """NaN -> None recursively (JSON has no NaN); mirrors orchestrator."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj
