"""Digest subpackage: turn eval artifacts into structured insight.

After every production run the eval stage leaves, under ``out_root``:

* ``reports/eval_{first}_{last}.json`` (+ ``.csv``) -- IC tables, Stage-1
  screen, walk-forward gates, permutation noise floors;
* ``reports/trial_ledger.jsonl`` -- the append-only honest-N ledger;
* ``parquet/dt={date}/factors.parquet`` -- the panel used for evaluation.

:func:`hft_autofactor.digest.build_digest` condenses those artifacts into
one JSON document plus a Chinese markdown insight report that drives the
next hypothesis round:

1. IC decay per factor across horizons + coarse half-life;
2. pass/fail taxonomy with hypothesized failure reasons;
3. correlation clusters among library factors (redundant families);
4. coverage gaps (unused panel information dimensions, weakest horizons);
5. data-quality notes (flag bits, ABSENT labels, one-sided book, NaN rates).

Everything is read-only against the pipeline outputs; the digest only ever
WRITES into its own report directory.
"""
from .correlations import greedy_clusters, pairwise_spearman
from .coverage import (
    DIMENSIONS,
    FACTOR_DIMENSIONS,
    coverage_report,
)
from .data_quality import (
    FLAG_BIT_NAMES,
    panel_quality,
    parquet_paths_for_dates,
    sample_factor_rows,
)
from .ic_decay import decay_table
from .report import (
    build_digest,
    find_eval_report,
    ledger_counts,
    render_markdown,
    write_digest,
)
from .taxonomy import classify_outcomes

__all__ = [
    "DIMENSIONS",
    "FACTOR_DIMENSIONS",
    "FLAG_BIT_NAMES",
    "build_digest",
    "classify_outcomes",
    "coverage_report",
    "decay_table",
    "find_eval_report",
    "greedy_clusters",
    "ledger_counts",
    "panel_quality",
    "parquet_paths_for_dates",
    "pairwise_spearman",
    "render_markdown",
    "sample_factor_rows",
    "write_digest",
]
