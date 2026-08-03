"""``hftaf-backtest`` command-line entry point (Stage 5).

Runs the cost-aware backtest of one admitted factor at one horizon under the
selected commission scenarios and writes reports under the backtest output
directory (default ``{out_root}/backtest/{factor}_h{horizon}/``):

* ``report.json``    -- full config echo, per-scenario metrics, gate verdict;
* ``summary.csv``    -- one row per scenario;
* ``per_day_{scenario}.csv`` and ``equity_{scenario}.csv``.

Usage::

    hftaf-backtest --config config/pipeline.yaml --factor oir --horizon 60 \
        --dates 20250101..20250601 \
        [--scenarios institutional,retail_negotiated,retail_default] \
        [--inventory 510300:100000] [--out /data/factor_lzt/backtest/...]

``--dates`` accepts either a comma-separated list of YYYYMMDD days or an
inclusive ``START..END`` range, which is expanded against the existing
parquet day partitions (``dt=YYYYMMDD``).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import polars as pl

from .costs import load_cost_models
from .engine import InstrumentMeta, run_backtest
from .metrics import gate_on_costs, summarize_results
from .signals import PositionRule

__all__ = ["main"]

_PARAMS_CANDIDATES = (
    Path("docs/knowledge/etf_backtest_params.yaml"),
    Path("docs/etf_backtest_params.yaml"),
)


def _find_params_yaml(config_path: Path) -> Path:
    candidates = list(_PARAMS_CANDIDATES)
    # Repo root inferred from the config file location (config/pipeline.yaml).
    root = Path(config_path).resolve().parent.parent
    candidates += [root / rel for rel in _PARAMS_CANDIDATES]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "etf_backtest_params.yaml not found (looked in docs/knowledge/ and "
        "docs/ relative to CWD and to the config file); pass --params-yaml"
    )


def _resolve_dates(parquet_dir: Path, spec: str) -> list[str]:
    """Expand ``--dates`` into a concrete list of YYYYMMDD days."""
    spec = spec.strip()
    if ".." in spec:
        start, _, end = spec.partition("..")
        start, end = start.strip(), end.strip()
        dates: list[str] = []
        if parquet_dir.is_dir():
            for p in sorted(parquet_dir.iterdir()):
                if p.is_dir() and p.name.startswith("dt="):
                    d = p.name[3:]
                    if start <= d <= end:
                        dates.append(d)
        if not dates:
            raise ValueError(
                f"no parquet partitions dt={start}..{end} under {parquet_dir} "
                "(run the convert stage first)"
            )
        return dates
    dates = [d.strip() for d in spec.split(",") if d.strip()]
    if not dates:
        raise ValueError("--dates is empty")
    return dates


def _parse_inventory(entries: Sequence[str]) -> dict[str, float]:
    inventory: dict[str, float] = {}
    for entry in entries:
        for chunk in entry.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(
                    f"--inventory entries are CODE:UNITS, got {chunk!r}"
                )
            code, _, units = chunk.partition(":")
            inventory[code.strip()] = float(units)
    return inventory


def _load_settlement_and_mechanics(params_yaml: Path) -> dict:
    """Read settlement category lists + trading mechanics from the params yaml.

    Returns a dict with keys: t_plus_0, t_plus_1, instrument_categories,
    tick_size_cny, lot_size_units, max_order_units, price_limit_pct.
    """
    import yaml

    with open(params_yaml, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    st = doc.get("settlement") or {}
    t_plus_0 = frozenset(st.get("t_plus_0") or [])
    t_plus_1 = frozenset(st.get("t_plus_1") or [])

    # Optional forward-compatible code -> category override table. Not present
    # in fee_table_v1 (built later from fund prospectuses); defaults to
    # equity_etf / T+1 when absent.
    cats = doc.get("instrument_categories") or {}
    instrument_categories = {str(k): str(v) for k, v in cats.items()}

    mech = doc.get("trading_mechanics") or {}
    limit = (mech.get("price_limit") or {}).get("pct", 0.10)
    return {
        "t_plus_0": t_plus_0,
        "t_plus_1": t_plus_1,
        "instrument_categories": instrument_categories,
        "tick_size_cny": float(mech.get("tick_size_cny", 0.001)),
        "lot_size_units": int(mech.get("lot_size_units", 100)),
        "max_order_units": int(mech.get("max_order_units", 1_000_000)),
        "price_limit_pct": float(limit),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hftaf-backtest",
        description="Cost-aware A-share ETF backtest for admitted factors.",
    )
    p.add_argument("--config", default="config/pipeline.yaml",
                   help="pipeline YAML config (default: %(default)s)")
    p.add_argument("--factor", required=True, help="factor column to backtest")
    p.add_argument("--horizon", required=True, type=int,
                   help="prediction horizon in seconds (15/30/60/300/900)")
    p.add_argument("--dates", required=True,
                   help="YYYYMMDD[,YYYYMMDD...] or START..END inclusive range")
    p.add_argument("--scenarios", default=None,
                   help="comma-separated commission scenarios "
                        "(default: from pipeline config)")
    p.add_argument("--instruments", default=None,
                   help="comma-separated instrument codes to restrict to")
    p.add_argument("--inventory", action="append", default=[],
                   help="initial 底仓 inventory CODE:UNITS (repeatable / "
                        "comma-separated)")
    p.add_argument("--params-yaml", default=None,
                   help="etf_backtest_params.yaml path (default: auto-discover)")
    p.add_argument("--conservative-microfees", action="store_true",
                   help="add the +0.3bp/side regulatory+transfer fee sensitivity")
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--direction", type=int, default=1, choices=(1, -1))
    p.add_argument("--max-units", type=int, default=100_000)
    p.add_argument("--signal-lag", type=int, default=1)
    p.add_argument("--z-window", type=int, default=100,
                   help="causal z-score window in rows")
    p.add_argument("--min-net-sharpe", type=float, default=0.5,
                   help="cost gate: minimum net annualized Sharpe per scenario")
    p.add_argument("--min-days", type=int, default=20,
                   help="cost gate: minimum backtest days per scenario")
    p.add_argument("--out", default=None,
                   help="output directory (default: {out_root}/backtest/"
                        "{factor}_h{horizon})")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Lazy imports: keep the backtest library importable without the rest of
    # the pipeline package being present.
    try:
        from hft_autofactor.config import load_config
        from hft_autofactor.ingest import load_panel
    except ImportError as exc:  # pragma: no cover
        print(f"error: cannot import pipeline package: {exc}")
        return 2

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"error: cannot load config {args.config}: {exc}")
        return 2

    try:
        dates = _resolve_dates(Path(cfg.out_root) / "parquet", args.dates)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2

    instruments = None
    if args.instruments:
        instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    try:
        panel = load_panel(cfg, dates, instruments=instruments, factors=[args.factor])
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2
    if panel.height == 0 or args.factor not in panel.columns:
        print(f"error: no rows (or factor column {args.factor!r} missing) for "
              f"{len(dates)} dates")
        return 2

    try:
        params_yaml = (
            Path(args.params_yaml) if args.params_yaml else _find_params_yaml(Path(args.config))
        )
        models = load_cost_models(
            params_yaml, conservative_microfees=args.conservative_microfees
        )
        settle = _load_settlement_and_mechanics(params_yaml)
        inventory = _parse_inventory(args.inventory)
    except Exception as exc:
        print(f"error: {exc}")
        return 2

    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    else:
        scenarios = list(cfg.commission_scenarios)
    unknown = [s for s in scenarios if s not in models]
    if unknown:
        print(f"error: unknown commission scenario(s): {unknown}; "
              f"available: {sorted(models)}")
        return 2

    # --- per-instrument meta: settlement class comes from the category map --
    t0, t1 = settle["t_plus_0"], settle["t_plus_1"]
    code_cat = settle["instrument_categories"]
    meta: dict[str, InstrumentMeta] = {}
    pairs = panel.select(["instrument", "exchange"]).unique()
    for inst_raw, exch_raw in pairs.iter_rows():
        inst, exch = str(inst_raw), str(exch_raw)
        category = code_cat.get(inst, "equity_etf")
        # Unknown categories default to the conservative side (T+1).
        settlement = "T+0" if category in t0 else "T+1"
        meta[inst] = InstrumentMeta(
            exchange=exch,
            settlement=settlement,
            etf_category=category,
            tick_cny=settle["tick_size_cny"],
            lot=settle["lot_size_units"],
            max_order_units=settle["max_order_units"],
            price_limit_pct=settle["price_limit_pct"],
        )

    rule = PositionRule(
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        direction=args.direction,
        max_position_units=args.max_units,
        signal_lag_rows=args.signal_lag,
    )

    results = {}
    for sc in scenarios:
        results[sc] = run_backtest(
            panel,
            args.factor,
            int(args.horizon),
            meta,
            models[sc],
            rule,
            dates=dates,
            initial_inventory_units=inventory or None,
            z_window_rows=args.z_window,
        )

    gate_ok, gate_details = gate_on_costs(
        results, min_net_sharpe=args.min_net_sharpe, min_days=args.min_days
    )
    summary = summarize_results(results)

    out_dir = (
        Path(args.out)
        if args.out
        else Path(cfg.out_root) / "backtest" / f"{args.factor}_h{args.horizon}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "factor": args.factor,
        "horizon_s": int(args.horizon),
        "dates": dates,
        "n_dates": len(dates),
        "scenarios": scenarios,
        "params_yaml": str(params_yaml),
        "conservative_microfees": bool(args.conservative_microfees),
        "rule": {
            "entry_z": args.entry_z,
            "exit_z": args.exit_z,
            "direction": args.direction,
            "max_position_units": args.max_units,
            "signal_lag_rows": args.signal_lag,
        },
        "z_window_rows": args.z_window,
        "initial_inventory_units": inventory,
        "instruments": sorted(meta.keys()),
        "gate": {"passed": gate_ok, **gate_details},
        "summary": summary.to_dicts(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary.write_csv(out_dir / "summary.csv")
    for sc, r in results.items():
        r.per_day.write_csv(out_dir / f"per_day_{sc}.csv")
        r.equity_curve.write_csv(out_dir / f"equity_{sc}.csv")

    # --- console summary -----------------------------------------------------
    print(f"factor={args.factor} horizon={args.horizon}s dates={len(dates)} "
          f"instruments={len(meta)}")
    for row in summary.to_dicts():
        print(
            "  {scenario:>18}: pnl={total_pnl_cny:>14.2f} CNY  "
            "sharpe={sharpe_annualized:>7.3f}  days={n_days:>4}  "
            "trades={n_trades:>6}  rt_cost={realized_round_trip_cost_bps:>6.2f}bp".format(**row)
        )
    verdict = "PASS" if gate_ok else "FAIL"
    print(f"cost gate ({args.min_net_sharpe} net Sharpe, {args.min_days} days): {verdict}")
    if not gate_ok:
        for sc, d in gate_details.get("scenarios", {}).items():
            for reason in d.get("reasons", []):
                print(f"  {sc}: {reason}")
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
