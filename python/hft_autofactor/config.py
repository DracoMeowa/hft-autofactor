"""Pipeline configuration: dataclass + YAML loader + derived path helpers.

All artifacts live under ``out_root`` (production: /data/factor_lzt). Nothing
is ever written into the read-only exchange data roots (/data/sse, /data/szse).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_HORIZONS_S = [15, 30, 60, 300, 900]
DEFAULT_COMMISSION_SCENARIOS = ["institutional", "retail_negotiated", "retail_default"]


@dataclass
class PipelineConfig:
    """Runtime configuration shared by every stage of the pipeline."""

    data_roots: dict[str, Path]                      # {"sse": ..., "szse": ...}
    out_root: Path                                   # /data/factor_lzt
    engine_bin: Path                                 # hftaf-engine executable
    horizons_s: list[int] = field(default_factory=lambda: list(DEFAULT_HORIZONS_S))
    factors: list[str] = field(default_factory=list)  # [] => full default registry
    max_workers: int = 2
    commission_scenarios: list[str] = field(
        default_factory=lambda: list(DEFAULT_COMMISSION_SCENARIOS)
    )

    # ------------------------------------------------------------------ #
    # Derived directory helpers (created lazily by ensure_dirs()).       #
    # ------------------------------------------------------------------ #
    @property
    def raw_dir(self) -> Path:
        return self.out_root / "raw"

    @property
    def parquet_dir(self) -> Path:
        return self.out_root / "parquet"

    @property
    def validation_dir(self) -> Path:
        return self.out_root / "validation"

    @property
    def reports_dir(self) -> Path:
        return self.out_root / "reports"

    @property
    def backtest_dir(self) -> Path:
        return self.out_root / "backtest"

    @property
    def logs_dir(self) -> Path:
        return self.out_root / "logs"

    @property
    def golden_dir(self) -> Path:
        return self.validation_dir / "golden"

    def raw_csv(self, date: str, exchange: str, channel: int) -> Path:
        """Interchange CSV path for one (date, exchange, channel) job."""
        return self.raw_dir / date / f"{exchange}_ch{channel}.csv"

    def parquet_path(self, date: str) -> Path:
        return self.parquet_dir / f"dt={date}" / "factors.parquet"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.parquet_dir,
            self.validation_dir,
            self.reports_dir,
            self.backtest_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _as_path(value: Any, default: str | None = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("missing required path in config")
        value = default
    return Path(str(value))


def load_config(path: str | Path = "config/pipeline.yaml") -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file.

    Expected (all keys optional except data_roots/out_root/engine_bin)::

        data_roots: {sse: /data/sse, szse: /data/szse}
        out_root: /data/factor_lzt
        engine_bin: build/cpp/hftaf-engine
        horizons: [15, 30, 60, 300, 900]
        factors: []                 # empty => all defaults
        max_workers: 2
        commission_scenarios: [institutional, retail_negotiated, retail_default]
    """
    cfg_path = Path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: Mapping[str, Any] = yaml.safe_load(fh) or {}

    roots_raw = raw.get("data_roots") or {}
    data_roots = {str(k): _as_path(v) for k, v in roots_raw.items()}

    horizons = raw.get("horizons") or raw.get("horizons_s") or DEFAULT_HORIZONS_S
    factors = raw.get("factors") or []
    scenarios = raw.get("commission_scenarios") or DEFAULT_COMMISSION_SCENARIOS

    return PipelineConfig(
        data_roots=data_roots,
        out_root=_as_path(raw.get("out_root")),
        engine_bin=_as_path(raw.get("engine_bin"), default="hftaf-engine"),
        horizons_s=[int(h) for h in horizons],
        factors=[str(f) for f in factors],
        max_workers=int(raw.get("max_workers", 2)),
        commission_scenarios=[str(s) for s in scenarios],
    )
