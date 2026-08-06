"""``hftaf-explore`` CLI: subcommands list | add | run | screen.

Minute-scale factor prototyping on the materialized parquet panel -- no C++
engine, no raw-data access.  Examples::

    hftaf-explore list   --config config/pipeline.yaml
    hftaf-explore add    --spec prototypes/my_idea.py
    hftaf-explore run    --dates 20250701..20250731 --protos my_idea --chunk-days 5
    hftaf-explore screen --dates 20250701..20250731 --protos my_idea

``run`` arms the panel prefix causality test: a prototype that fails any
truncation cutoff is rejected and its partitions are removed.  ``screen``
applies the RankIC/NW + library-dedup + purged IS/OOS pre-screen.

Exit codes: 0 ok, 1 at least one rejection/operational failure, 2 usage or
config error.  Installed as the ``hftaf-explore`` console script
(``pyproject.toml``); ``python -m hft_autofactor.explore`` is equivalent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ..config import load_config
from ..pipeline.cli import expand_date_spec
from .layout import prototypes_dir, spec_meta_path, spec_path
from .registry import (
    Prototype,
    PrototypeError,
    PrototypeRegistry,
    default_registry,
    load_prototype_spec,
)
from .runner import run_prototype
from .screen import PANEL_FACTORS, ScreenConfig, screen_prototype

__all__ = ["main", "build_parser", "load_registry"]


def load_registry(cfg) -> PrototypeRegistry:
    """Built-ins + persisted specs under ``{out_root}/explore/prototypes``.

    A persisted spec colliding with a built-in name is skipped with a
    warning (built-ins win; re-adding under a taken name is refused).
    """
    registry = default_registry()
    spec_dir = prototypes_dir(cfg)
    if spec_dir.is_dir():
        for path in sorted(spec_dir.glob("*.py")):
            try:
                proto = load_prototype_spec(path, source=str(path))
            except PrototypeError as exc:
                print(f"warning: skipping spec {path}: {exc}", file=sys.stderr)
                continue
            if proto.name in registry:
                print(
                    f"warning: persisted spec {proto.name!r} collides with an "
                    "existing registration; skipped",
                    file=sys.stderr,
                )
                continue
            registry.register(proto)
    return registry


def _parse_int_list(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hftaf-explore",
        description="explore lane: minute-scale factor prototyping on the "
        "materialized parquet panel (no C++, no raw data)",
    )
    parser.add_argument(
        "--config", default="config/pipeline.yaml", help="pipeline YAML config"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered prototypes")

    p_add = sub.add_parser("add", help="validate + persist a prototype spec file")
    p_add.add_argument("--spec", required=True, help="path to a PROTOTYPE spec .py")
    p_add.add_argument(
        "--overwrite", action="store_true",
        help="replace an already-persisted prototype of the same name",
    )

    p_run = sub.add_parser(
        "run", help="compute prototypes over a date range (causality-gated)"
    )
    p_run.add_argument("--dates", required=True, help="YYYYMMDD[,YYYYMMDD|A..B]")
    p_run.add_argument(
        "--protos", default=None,
        help="comma list of prototype names (default: all registered)",
    )
    p_run.add_argument("--chunk-days", type=int, default=5)
    p_run.add_argument("--k", type=int, default=8, help="truncation cutoffs per chunk")
    p_run.add_argument("--overwrite", action="store_true")

    p_scr = sub.add_parser(
        "screen", help="RankIC/NW + library dedup + purged IS/OOS pre-screen"
    )
    p_scr.add_argument("--dates", required=True, help="YYYYMMDD[,YYYYMMDD|A..B]")
    p_scr.add_argument(
        "--protos", default=None,
        help="comma list of prototype names (default: all registered)",
    )
    p_scr.add_argument("--horizons", default=None, help="comma list in seconds")
    p_scr.add_argument(
        "--library-factors", default=None,
        help="dedup universe for the duplication gate: comma list of panel "
        "column names, or the literal 'panel' = every factor column present "
        "in the panel (default: the 12 canonical library factors)",
    )
    p_scr.add_argument("--max-abs-corr", type=float, default=0.85)
    p_scr.add_argument("--min-is-t", type=float, default=2.0)
    p_scr.add_argument("--min-oos-t", type=float, default=2.0)
    p_scr.add_argument("--min-retention", type=float, default=0.5)
    p_scr.add_argument("--embargo-days", type=int, default=1)
    p_scr.add_argument("--n-test-days", type=int, default=5)

    return parser


def _select_protos(
    registry: PrototypeRegistry, spec: str | None
) -> list[Prototype]:
    if not spec:
        return list(registry)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    return [registry.get(n) for n in names]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load config {args.config!r}: {exc}", file=sys.stderr)
        return 2

    registry = load_registry(cfg)

    # --------------------------- list ------------------------------- #
    if args.command == "list":
        if len(registry) == 0:
            print("(no prototypes registered)")
            return 0
        for proto in registry:
            mech = proto.mechanism.replace("\n", " ")
            if len(mech) > 72:
                mech = mech[:69] + "..."
            print(f"{proto.name:<24} [{proto.source}]")
            print(f"    mechanism:   {mech}")
            print(f"    info set:    {proto.info_set}")
            print(f"    inspiration: {proto.inspiration}")
        print(f"{len(registry)} prototype(s)")
        return 0

    # ---------------------------- add ------------------------------- #
    if args.command == "add":
        try:
            proto = load_prototype_spec(args.spec, source=str(Path(args.spec)))
        except PrototypeError as exc:
            print(f"error: refused: {exc}", file=sys.stderr)
            return 1
        dst = spec_path(cfg, proto.name)
        if dst.is_file() and not args.overwrite:
            print(
                f"error: prototype {proto.name!r} already persisted at {dst} "
                "(use --overwrite to replace)",
                file=sys.stderr,
            )
            return 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(args.spec).read_bytes())
        meta = spec_meta_path(cfg, proto.name)
        meta.write_text(
            json.dumps(proto.metadata_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"added {proto.name} -> {dst}")
        return 0

    # ------------------------- run / screen -------------------------- #
    try:
        dates = expand_date_spec(args.dates)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not dates:
        print("error: no dates given", file=sys.stderr)
        return 2

    try:
        protos = _select_protos(registry, args.protos)
    except PrototypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not protos:
        print("error: no prototypes selected", file=sys.stderr)
        return 2

    if args.command == "run":
        n_bad = 0
        for proto in protos:
            result = run_prototype(
                cfg, proto, dates,
                chunk_days=args.chunk_days, k=args.k, overwrite=args.overwrite,
            )
            if result.status == "ok":
                print(
                    f"{'OK':>18}  {proto.name:<24} {len(result.partitions)} "
                    f"partition(s)  {result.report_path}"
                )
            elif result.status == "skipped":
                print(
                    f"{'SKIPPED':>18}  {proto.name:<24} up-to-date "
                    f"({len(result.partitions)} partition(s))"
                )
            elif result.status == "rejected_causality":
                n_bad += 1
                print(f"{'REJECTED':>18}  {proto.name:<24} causality: {result.message}")
                print(f"{'':>18}  report: {result.report_path}")
            else:
                n_bad += 1
                print(f"{'FAILED':>18}  {proto.name:<24} {result.message}")
                if result.report_path:
                    print(f"{'':>18}  report: {result.report_path}")
        print(
            f"{len(protos)} prototype(s): {len(protos) - n_bad} ok/skipped, "
            f"{n_bad} rejected/failed"
        )
        return 1 if n_bad else 0

    if args.command == "screen":
        horizons = _parse_int_list(args.horizons)
        if args.library_factors is None:
            library_factors: object = None
        elif args.library_factors.strip().lower() == "panel":
            library_factors = PANEL_FACTORS
        else:
            library_factors = [
                n.strip()
                for n in args.library_factors.split(",")
                if n.strip()
            ]
        sc = ScreenConfig(
            max_abs_corr=args.max_abs_corr,
            min_is_t=args.min_is_t,
            min_oos_t=args.min_oos_t,
            min_retention=args.min_retention,
            embargo_days=args.embargo_days,
            n_test_days=args.n_test_days,
        )
        n_bad = 0
        for proto in protos:
            try:
                report = screen_prototype(
                    cfg, proto, dates, horizons=horizons, screen_cfg=sc,
                    library_factors=library_factors,
                )
            except (FileNotFoundError, ValueError, KeyError) as exc:
                n_bad += 1
                print(f"{'ERROR':>10}  {proto.name:<24} {exc}")
                continue
            verdict = "PASS" if report.passed else "FAIL"
            print(
                f"{verdict:>10}  {proto.name:<24} status={report.status}  "
                f"max|corr|={report.duplicate_check.get('max_abs_corr', float('nan')):.3f}"
            )
            for reason in report.reasons:
                print(f"{'':>10}    - {reason}")
            for row in report.horizons:
                gate = "pass" if row["passed"] else "fail"
                print(
                    f"{'':>10}    h={row['horizon_s']:>4}s  "
                    f"is_ic={row['is_mean_ic']:+.4f} (t={row['is_t_stat_nw']:+.2f})  "
                    f"oos_ic={row['oos_mean_ic']:+.4f} (t={row['oos_t_stat_nw']:+.2f})  "
                    f"[{gate}]"
                )
            if report.report_path:
                print(f"{'':>10}    report: {report.report_path}")
        return 1 if n_bad else 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
