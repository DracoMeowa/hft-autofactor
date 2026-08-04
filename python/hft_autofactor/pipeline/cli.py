"""hftaf CLI: subcommands factors | convert | eval | mask.

Examples::

    hftaf factors  --config config/pipeline.yaml --dates 20250603,20250604
    hftaf convert  --dates 20250603..20250607 --overwrite
    hftaf eval     --dates 20250603..20250630
    hftaf mask     --dates 20250603 --k 4

Single-instrument pilot (e.g. 588000)::

    hftaf convert  --dates 20250701..20250930 --instruments 588000
    hftaf eval     --dates 20250701..20250930 --instruments 588000
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from typing import Sequence

from ..config import load_config
from . import orchestrator


def _validate_date(s: str) -> str:
    s = s.strip()
    if len(s) != 8 or not s.isdigit():
        raise argparse.ArgumentTypeError(f"bad date {s!r} (want YYYYMMDD)")
    try:
        date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad date {s!r}: {exc}") from exc
    return s


def expand_date_spec(spec: str) -> list[str]:
    """Expand ``A,B,..,C..D`` into a flat, de-duplicated date list."""
    out: list[str] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ".." in item:
            a, b = (x.strip() for x in item.split("..", 1))
            _validate_date(a)
            _validate_date(b)
            d0 = date(int(a[0:4]), int(a[4:6]), int(a[6:8]))
            d1 = date(int(b[0:4]), int(b[4:6]), int(b[6:8]))
            if d1 < d0:
                raise ValueError(f"empty date range {item!r}")
            while d0 <= d1:
                out.append(d0.strftime("%Y%m%d"))
                d0 += timedelta(days=1)
        else:
            out.append(_validate_date(item))
    return list(dict.fromkeys(out))


def _parse_int_list(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_str_list(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    return [x.strip() for x in spec.split(",") if x.strip()] or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hftaf",
        description="hft-autofactor evaluation pipeline (stages 1-2-3-4)",
    )
    parser.add_argument(
        "--config", default="config/pipeline.yaml", help="pipeline YAML config"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_fac = sub.add_parser("factors", help="run the C++ factor/label engine")
    p_fac.add_argument("--dates", required=True, help="YYYYMMDD[,YYYYMMDD|A..B]")
    p_fac.add_argument("--channels", default=None, help="comma list, e.g. 1,2,3")
    p_fac.add_argument("--workers", type=int, default=None)
    p_fac.add_argument("--overwrite", action="store_true")
    p_fac.add_argument("--dry-run", action="store_true")

    p_conv = sub.add_parser("convert", help="raw CSV -> day parquet partitions")
    p_conv.add_argument("--dates", required=True)
    p_conv.add_argument("--overwrite", action="store_true")
    p_conv.add_argument(
        "--instruments", default=None,
        help="comma list of instrument codes to keep (default: all). "
             "Recorded in the partition sidecar so a filtered parquet is "
             "never mistaken for a full one.",
    )

    p_eval = sub.add_parser("eval", help="IC/RankIC evaluation + gating report")
    p_eval.add_argument("--dates", required=True)
    p_eval.add_argument("--factors", default=None, help="comma list (default: all)")
    p_eval.add_argument("--horizons", default=None, help="comma list in seconds")
    p_eval.add_argument(
        "--instruments", default=None,
        help="comma list of instrument codes to evaluate (default: all). "
             "With a single instrument, cross-sectional IC is skipped and "
             "all gating uses the time-series RankIC.",
    )

    p_mask = sub.add_parser("mask", help="lookahead mask validation (Stage 2)")
    p_mask.add_argument("--dates", required=True)
    p_mask.add_argument("--k", type=int, default=4, help="truncation points per job")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load config {args.config!r}: {exc}", file=sys.stderr)
        return 2

    try:
        dates = expand_date_spec(args.dates)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not dates:
        print("error: no dates given", file=sys.stderr)
        return 2

    if args.command == "factors":
        channels = _parse_int_list(args.channels)
        results = orchestrator.run_factor_stage(
            cfg,
            dates,
            channels=channels,
            max_workers=args.workers,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        n_bad = 0
        for r in results:
            print(
                f"{r.status:>8}  {r.job.date} {r.job.exchange} "
                f"ch{r.job.channel}  rc={r.returncode}  {r.elapsed_s:.1f}s"
            )
            if r.status == "failed":
                n_bad += 1
                if r.log_tail:
                    print(f"          tail: {r.log_tail.strip()[-300:]}")
        print(f"{len(results)} jobs: {len(results) - n_bad} ok/skipped, {n_bad} failed")
        return 1 if n_bad else 0

    if args.command == "convert":
        try:
            paths = orchestrator.run_convert_stage(
                cfg,
                dates,
                overwrite=args.overwrite,
                instruments=_parse_str_list(args.instruments),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for p in paths:
            print(p)
        return 0

    if args.command == "eval":
        factor_list = (
            [f.strip() for f in args.factors.split(",") if f.strip()]
            if args.factors
            else None
        )
        horizons = _parse_int_list(args.horizons)
        try:
            report = orchestrator.run_eval_stage(
                cfg,
                dates,
                factors=factor_list,
                horizons=horizons,
                instruments=_parse_str_list(args.instruments),
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(report)
        return 0

    if args.command == "mask":
        report = orchestrator.run_mask_stage(cfg, dates, k=args.k)
        import json

        summary = json.loads(report.read_text(encoding="utf-8"))
        print(
            f"mask validation: {summary['n_passed']}/{summary['n_jobs']} jobs passed"
        )
        for e in summary["entries"]:
            state = "ERROR" if e["error"] else (
                "PASS" if e["report"]["passed"] else "FAIL"
            )
            print(
                f"{state:>5}  {e['date']} {e['exchange']} ch{e['channel']}"
                + (f"  ({e['error']})" if e["error"] else "")
            )
        print(report)
        return 0 if summary["n_passed"] == summary["n_jobs"] else 1

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
