"""Prototype runner: chunked panel compute + prefix causality gate + output.

Flow of :func:`run_prototype` for one prototype over a date range:

  1. day partitions are grouped into chunks of ``chunk_days`` consecutive
     days (whole days only -- ``(date, instrument)`` groups are never split,
     so chunked compute is bit-identical to full-range compute);
  2. each chunk is loaded via ``ingest.load_panel``, and the panel prefix
     causality test runs on it; a prototype that fails ANY cutoff is
     rejected and no partitions are kept;
  3. passing chunks get their prototype column written to
     ``{out_root}/explore/panels/{name}/dt={date}.parquet`` (skip-if-done
     unless ``overwrite``).

Label columns are stripped before the compute spec runs, so a prototype can
never read its own targets; the info set is limited to base columns +
library factor columns.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import polars as pl

from ..config import PipelineConfig
from ..ingest import LABEL_PREFIXES, load_panel
from .causality import PanelCausalityReport, panel_prefix_check
from .layout import panel_path, run_report_path, sanitize_for_json
from .registry import Prototype

__all__ = [
    "PrototypeComputeError",
    "RunResult",
    "compute_prototype_column",
    "run_prototype",
    "load_explore_panel",
]

_GROUP_COLS = ("date", "instrument")
_ROW_COL = "__hftaf_explore_row"


class PrototypeComputeError(RuntimeError):
    """A compute spec crashed or broke its row-alignment contract."""


def _label_columns(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if any(c.startswith(p) for p in LABEL_PREFIXES)]


def _normalize_values(proto: Prototype, values: object, n_rows: int) -> pl.Series:
    """Coerce a compute-spec return value into an aligned Float64 Series."""
    if isinstance(values, pl.DataFrame):
        if values.width != 1:
            raise PrototypeComputeError(
                f"prototype {proto.name!r}: compute returned a DataFrame with "
                f"{values.width} columns; return exactly one column"
            )
        series = values.to_series()
    elif isinstance(values, pl.Series):
        series = values
    elif isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise PrototypeComputeError(
                f"prototype {proto.name!r}: compute returned a "
                f"{values.ndim}-d array; want 1-d"
            )
        series = pl.Series(proto.name, values)
    else:
        raise PrototypeComputeError(
            f"prototype {proto.name!r}: compute returned "
            f"{type(values).__name__}; want pl.Series / np.ndarray / 1-col pl.DataFrame"
        )
    if len(series) != n_rows:
        raise PrototypeComputeError(
            f"prototype {proto.name!r}: compute returned {len(series)} values "
            f"for a group of {n_rows} rows (must align row-for-row)"
        )
    return series.cast(pl.Float64).rename(proto.name)


def compute_prototype_column(panel: pl.DataFrame, proto: Prototype) -> pl.DataFrame:
    """Append the prototype's column to ``panel`` (input row order preserved).

    Computation is per ``(date, instrument)`` group, each handed to the
    compute spec sorted by ``ts_ms`` ascending WITHOUT any label columns
    (``fwd_*_ret_*``): a prototype's info set can never include the targets.
    No cross-day state is ever carried (same contract as
    ``backtest.signals.zscore_column``).
    """
    if proto.name in panel.columns:
        raise PrototypeComputeError(
            f"panel already contains column {proto.name!r}; refusing to shadow it"
        )
    if panel.height == 0:
        return panel.with_columns(
            pl.Series(proto.name, [], dtype=pl.Float64)
        )
    for col in ("date", "instrument", "ts_ms"):
        if col not in panel.columns:
            raise PrototypeComputeError(
                f"panel lacks grouping column {col!r}; have {panel.columns}"
            )

    drop = set(_label_columns(panel.columns)) | {_ROW_COL}
    spec_cols = [c for c in panel.columns if c not in drop]

    indexed = panel.with_row_index(_ROW_COL)
    ordered = indexed.sort([*_GROUP_COLS, "ts_ms"])

    parts: list[pl.DataFrame] = []
    for part in ordered.partition_by(list(_GROUP_COLS), maintain_order=True):
        values = proto.compute(part.select(spec_cols))
        series = _normalize_values(proto, values, part.height)
        parts.append(part.select(_ROW_COL).with_columns(series))

    computed = pl.concat(parts).sort(_ROW_COL).drop(_ROW_COL)
    return indexed.drop(_ROW_COL).with_columns(computed[proto.name])


def _chunks(dates: Sequence[str], chunk_days: int) -> list[list[str]]:
    ordered = sorted(set(dates))
    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    return [
        ordered[i : i + chunk_days] for i in range(0, len(ordered), chunk_days)
    ]


@dataclass
class RunResult:
    prototype: str
    status: str  # "ok" | "skipped" | "rejected_causality" | "failed"
    partitions: list[Path] = field(default_factory=list)
    causality: PanelCausalityReport | None = None
    report_path: Path | None = None
    message: str = ""


def run_prototype(
    cfg: PipelineConfig,
    proto: Prototype,
    dates: Sequence[str],
    *,
    chunk_days: int = 5,
    k: int = 8,
    overwrite: bool = False,
) -> RunResult:
    """Compute one prototype over ``dates`` with the causality gate armed.

    Returns a :class:`RunResult`; on causality failure the prototype is
    rejected, every partition written by THIS run is removed, and the
    rejection is recorded in the run report JSON.
    """
    dates_sorted = sorted(set(dates))
    if not dates_sorted:
        return RunResult(proto.name, "failed", message="no dates given")

    targets = [panel_path(cfg, proto.name, d) for d in dates_sorted]
    if not overwrite and all(p.is_file() for p in targets):
        return RunResult(
            proto.name, "skipped", partitions=targets,
            message="all partitions up-to-date",
        )

    written: list[Path] = []   # partitions written by THIS run (cleaned up on reject)
    kept: list[Path] = []      # pre-existing partitions reused (never deleted)
    causality_report: PanelCausalityReport | None = None
    try:
        for chunk in _chunks(dates_sorted, chunk_days):
            panel = load_panel(cfg, chunk)
            if panel.is_empty():
                raise FileNotFoundError(f"chunk {chunk} loaded empty")
            report = panel_prefix_check(panel, proto, k=k)
            causality_report = report
            if not report.passed:
                first_bad = next(
                    i for i, d in enumerate(report.diffs) if not d.identical
                )
                cut = report.points[first_bad]
                raise _CausalityRejection(
                    f"prefix identity failed at cutoff {cut.label} "
                    f"({cut.date} ts_ms={cut.ts_ms}): "
                    f"{report.diffs[first_bad].first_diff}"
                )
            aug = compute_prototype_column(panel, proto)
            for date in chunk:
                out = panel_path(cfg, proto.name, date)
                if out.is_file() and not overwrite:
                    kept.append(out)  # kept from a previous run
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                aug.filter(pl.col("date") == date).write_parquet(out)
                written.append(out)
    except _CausalityRejection as exc:
        # reject: remove every partition THIS run wrote, keep report
        for path in written:
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        report_path = _write_run_report(
            cfg, proto, dates_sorted, "rejected_causality", str(exc),
            causality_report, [str(p) for p in written],
        )
        return RunResult(
            proto.name, "rejected_causality", causality=causality_report,
            report_path=report_path, message=str(exc),
        )
    except Exception as exc:  # operational failure: bubble details, no cleanup
        report_path = _write_run_report(
            cfg, proto, dates_sorted, "failed", f"{type(exc).__name__}: {exc}",
            causality_report, [str(p) for p in written],
        )
        return RunResult(
            proto.name, "failed", causality=causality_report,
            report_path=report_path, message=f"{type(exc).__name__}: {exc}",
        )

    partitions = [panel_path(cfg, proto.name, d) for d in dates_sorted]
    report_path = _write_run_report(
        cfg, proto, dates_sorted, "ok", "", causality_report,
        [str(p) for p in partitions],
    )
    return RunResult(
        proto.name, "ok", partitions=partitions,
        causality=causality_report, report_path=report_path,
    )


class _CausalityRejection(Exception):
    """Internal: prototype failed the panel prefix causality test."""


def _write_run_report(
    cfg: PipelineConfig,
    proto: Prototype,
    dates: list[str],
    status: str,
    message: str,
    causality: PanelCausalityReport | None,
    partitions: list[str],
) -> Path:
    path = run_report_path(cfg, proto.name, dates)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prototype": proto.metadata_dict(),
        "status": status,
        "message": message,
        "dates": dates,
        "partitions": partitions,
        "causality": (
            {
                "points": [dataclasses.asdict(p) for p in causality.points],
                "diffs": [dataclasses.asdict(d) for d in causality.diffs],
                "passed": causality.passed,
            }
            if causality is not None
            else None
        ),
    }
    path.write_text(
        json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_explore_panel(
    cfg: PipelineConfig, name: str, dates: Sequence[str]
) -> pl.DataFrame:
    """Load a prototype's augmented partitions for ``dates`` as one panel."""
    paths = [panel_path(cfg, name, d) for d in dates]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing explore partitions (run `hftaf-explore run` first): "
            + ", ".join(missing)
        )
    return pl.concat(
        [pl.read_parquet(p) for p in paths], how="vertical_relaxed"
    ).sort(["date", "instrument", "ts_ms"])
