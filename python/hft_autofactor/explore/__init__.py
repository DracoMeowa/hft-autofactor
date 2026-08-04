"""Explore lane: minute-scale factor prototyping on the materialized panel.

This subpackage operates ONLY on the parquet panel partitions produced by
the convert stage (``{out_root}/parquet/dt=YYYYMMDD/factors.parquet``) and
writes ONLY under ``{out_root}/explore``.  It never touches the raw exchange
dumps (/data/sse, /data/szse) and never invokes the C++ engine: a prototype
is a causal polars/numpy expression over panel columns, screened in minutes.

Modules:
  registry  -- Prototype dataclass + metadata-complete registration
  causality -- panel truncate-and-recompute prefix identity test
  runner    -- chunked compute + per-day partition writer
  screen    -- RankIC/NW pre-screen, library dedup, purged IS/OOS retention
  cli       -- ``hftaf-explore`` (list | add | run | screen)
"""
from .registry import (
    Prototype,
    PrototypeError,
    PrototypeRegistry,
    default_registry,
    explore_prototype,
    load_prototype_spec,
)
from .causality import (
    PanelCausalityReport,
    PanelCutoff,
    PanelPrefixDiff,
    choose_panel_cutoffs,
    compare_panel_prefix,
    panel_prefix_check,
)
from .runner import (
    RunResult,
    compute_prototype_column,
    load_explore_panel,
    run_prototype,
)
from .screen import (
    ScreenConfig,
    ScreenReport,
    library_correlations,
    screen_prototype,
)
from .layout import (
    explore_root,
    panel_path,
    panels_dir,
    prototypes_dir,
    reports_dir,
    run_report_path,
    screen_report_path,
    spec_path,
)

__all__ = [
    # registry
    "Prototype",
    "PrototypeError",
    "PrototypeRegistry",
    "default_registry",
    "explore_prototype",
    "load_prototype_spec",
    # causality
    "PanelCausalityReport",
    "PanelCutoff",
    "PanelPrefixDiff",
    "choose_panel_cutoffs",
    "compare_panel_prefix",
    "panel_prefix_check",
    # runner
    "RunResult",
    "compute_prototype_column",
    "load_explore_panel",
    "run_prototype",
    # screen
    "ScreenConfig",
    "ScreenReport",
    "library_correlations",
    "screen_prototype",
    # layout
    "explore_root",
    "panel_path",
    "panels_dir",
    "prototypes_dir",
    "reports_dir",
    "run_report_path",
    "screen_report_path",
    "spec_path",
]
