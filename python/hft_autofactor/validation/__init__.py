"""Validation subpackage: lookahead mask tests and golden-hash bookkeeping."""
from .golden import hash_output_csv, load_golden, store_golden
from .mask_test import (
    MaskReport,
    PrefixDiff,
    TruncationPoint,
    choose_truncation_points,
    compare_prefix,
    mask_test_day,
    run_engine,
    truncate_snapshot_file,
    truncate_tick_file,
)

__all__ = [
    "MaskReport",
    "PrefixDiff",
    "TruncationPoint",
    "choose_truncation_points",
    "compare_prefix",
    "hash_output_csv",
    "load_golden",
    "mask_test_day",
    "run_engine",
    "store_golden",
    "truncate_snapshot_file",
    "truncate_tick_file",
]
