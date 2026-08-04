"""``hftaf-digest`` command-line entry point (post-eval feedback digest).

Turns the eval-stage artifacts under an ``out_root`` (IC tables, gating
report, trial ledger) plus the parquet panel into one JSON document and one
Chinese markdown insight report for the next hypothesis round.

Usage::

    hftaf-digest --out-root /data/factor_lzt [--report-dir DIR]
                 [--dates 20250701..20250930] [--eval-report PATH]
                 [--max-rows 200000] [--corr-threshold 0.7] [--no-panel]

Defaults: the newest ``reports/eval_*.json`` is digested; the panel dates
come from that report; the digest is written to
``{out_root}/reports/digest/``.  The digest only WRITES into the report
directory -- never into the exchange data roots.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .correlations import DEFAULT_CORR_THRESHOLD
from .report import build_digest, write_digest

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hftaf-digest",
        description="Post-eval feedback digest: eval artifacts -> JSON + "
                    "Chinese markdown insight report.",
    )
    p.add_argument(
        "--out-root", required=True,
        help="pipeline output root (e.g. /data/factor_lzt)",
    )
    p.add_argument(
        "--report-dir", default=None,
        help="where to write digest_*.json/.md "
             "(default: {out_root}/reports/digest)",
    )
    p.add_argument(
        "--dates", default=None,
        help="YYYYMMDD[,YYYYMMDD|A..B] panel dates (default: the eval "
             "report's dates)",
    )
    p.add_argument(
        "--eval-report", default=None,
        help="explicit eval JSON path (default: newest reports/eval_*.json)",
    )
    p.add_argument(
        "--max-rows", type=int, default=200_000,
        help="max stride-sampled rows for the correlation pass "
             "(default: %(default)s)",
    )
    p.add_argument(
        "--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD,
        help="|corr| redundancy-cluster threshold (default: %(default)s)",
    )
    p.add_argument(
        "--no-panel", action="store_true",
        help="skip parquet panel statistics (eval-artifact-only digest)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    dates = None
    if args.dates:
        # reuse the hftaf date-spec grammar (A,B,..,C..D)
        from ..pipeline.cli import expand_date_spec

        try:
            dates = expand_date_spec(args.dates)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if not dates:
            print("error: --dates is empty", file=sys.stderr)
            return 2

    try:
        digest = build_digest(
            args.out_root,
            dates=dates,
            eval_report=args.eval_report,
            max_rows=args.max_rows,
            corr_threshold=args.corr_threshold,
            include_panel=not args.no_panel,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: digest failed: {exc}", file=sys.stderr)
        return 1

    from pathlib import Path

    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else Path(args.out_root) / "reports" / "digest"
    )
    try:
        json_path, md_path = write_digest(digest, report_dir)
    except OSError as exc:
        print(f"error: cannot write digest: {exc}", file=sys.stderr)
        return 1

    n_combos = len(digest.get("taxonomy", []))
    n_pass = sum(1 for r in digest.get("taxonomy", []) if r.get("combined_passed"))
    print(f"digest: {n_pass}/{n_combos} combos passed stage1 + walk-forward")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
