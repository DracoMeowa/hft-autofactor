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

``--panel-dir DIR`` loads the panel from an archived directory of day
partitions (``dt=YYYYMMDD.parquet`` files or ``dt=YYYYMMDD/`` dirs) instead
of the pipeline parquet store -- the route for explore-lane prototype factors
whose column lives in the explore panel, not the production parquet
(spec section 7 "因子列来源: explore panel join").

Track-B mode (#86, ``--matrix``)
--------------------------------
Instead of the full position simulation, runs the frozen 24-cell conditional
profitability matrix of ``docs/design/eval-redesign-86.md``: direction x
entry-threshold tau x volatility regime, non-overlapping carry-free trades,
net of the taker cost stack and (for shorts) the securities-lending borrow
cost.  ``--direction`` supplies the track-A ``ic_direction`` and fixes the
primary cell; ``--eval-dates`` restricts trading to a window inside
``--dates`` (rows outside it feed the trailing regime history only).
Reports go to ``--matrix-out`` (default ``{out_root}/matrix/{factor}_h{H}/``)
and every cell is appended to the trial ledger at stage="matrix_cell" before
the primary gate is read.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import polars as pl

from .conditional import MatrixConfig, run_conditional_matrix, write_matrix_report
from .costs import load_cost_models, load_short_cost_model
from .derived import DERIVED_FACTORS, materialize_derived
from .engine import InstrumentMeta, run_backtest
from .metrics import gate_on_costs, summarize_results
from .signals import PositionRule
from ..eval.gating import TrialLedger

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
                if not p.name.startswith("dt="):
                    continue
                d = p.name[3:]
                if d.endswith(".parquet"):
                    d = d[: -len(".parquet")]
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


def _run_matrix_mode(
    args,
    cfg,
    panel: pl.DataFrame,
    dates: list[str],
    scenarios: list[str],
    models: dict,
    settle: dict,
    params_yaml: Path,
) -> int:
    """Track-B conditional profitability matrix (#86).

    Frozen 24-cell set, primary-cell-only admission gate; every cell is
    ledgered at stage="matrix_cell" before the gate reads any threshold.
    """
    scen_models = {s: models[s] for s in scenarios}
    try:
        short_costs = load_short_cost_model(params_yaml)
    except Exception as exc:
        print(f"error: {exc}")
        return 2

    eval_dates: list[str] | None = None
    if args.eval_dates:
        partition_root = (
            Path(args.panel_dir)
            if args.panel_dir
            else Path(cfg.out_root) / "parquet"
        )
        try:
            eval_dates = _resolve_dates(partition_root, args.eval_dates)
        except ValueError as exc:
            print(f"error: --eval-dates: {exc}")
            return 2
        outside = [d for d in eval_dates if d not in set(dates)]
        if outside:
            print(f"error: --eval-dates outside --dates: {outside[:5]}")
            return 2

    ledger = TrialLedger(cfg.reports_dir / "trial_ledger.jsonl")
    result = run_conditional_matrix(
        panel,
        args.factor,
        int(args.horizon),
        scen_models,
        MatrixConfig(),
        ic_direction=int(args.direction),
        eval_dates=eval_dates,
        etf_categories=settle["instrument_categories"],
        short_costs=short_costs,
        ledger=ledger,
    )
    out_dir = (
        Path(args.matrix_out)
        if args.matrix_out
        else Path(cfg.out_root) / "matrix" / f"{args.factor}_h{args.horizon}"
    )
    report_path = write_matrix_report(out_dir, result)

    print(
        f"matrix factor={args.factor} horizon={args.horizon}s "
        f"ic_direction={result.ic_direction:+d} primary="
        f"{result.primary_cell['direction']}/tau={result.primary_cell['tau']}/"
        f"{result.primary_cell['regime']}"
    )
    borrow_note = (
        "none (shorts descriptive-only)"
        if short_costs is None
        else "%s/yr min %sd" % (short_costs.borrow_rate_annual, short_costs.min_charge_days)
    )
    print(
        f"  trades={result.trades.height}  cells={result.cells.height}  "
        f"borrow_model={borrow_note}"
    )
    for scen, v in result.primary.get("scenarios", {}).items():
        verdict = "PASS" if v["pass"] else "FAIL"
        print(
            "  %18s: %s  net=%.3fbp  t=%.2f  n=%d/%dd"
            % (scen, verdict, v["mean_net_edge_bps"], v["t_nw_daily"],
               v["n_trades"], v["n_days"])
        )
        for r in v.get("reasons", []):
            print(f"    - {r}")
    print(f"primary gate: {'PASS' if result.primary.get('passed') else 'FAIL'}")
    print(f"report: {report_path}")
    return 0


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
    p.add_argument("--direction", type=int, default=1, choices=(1, -1),
                   help="position direction (full sim); in --matrix mode this "
                        "is the track-A ic_direction = sign(mean IC) and fixes "
                        "the primary cell (1 = long primary, -1 = short/"
                        "reversal primary)")
    p.add_argument("--max-units", type=int, default=100_000)
    p.add_argument("--base-units", type=int, default=0,
                   help="inventory floor (底仓): targets oscillate around "
                        "this many units; pair with matching --inventory to "
                        "enable T+1 intraday round trips (底仓做T). "
                        "0 = plain long/flat")
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
    p.add_argument("--matrix", action="store_true",
                   help="run the track-B conditional profitability matrix "
                        "(#86) instead of the full position simulation")
    p.add_argument("--matrix-out", default=None,
                   help="matrix report directory (default: {out_root}/matrix/"
                        "{factor}_h{horizon})")
    p.add_argument("--eval-dates", default=None,
                   help="matrix mode: trading window inside --dates "
                        "(same syntax); rows outside it feed the trailing "
                        "regime history only. Default: all --dates")
    p.add_argument("--panel-dir", default=None,
                   help="load the panel from an archived directory of day "
                        "partitions (dt=YYYYMMDD.parquet files or "
                        "dt=YYYYMMDD/ dirs) instead of the pipeline parquet "
                        "store -- for explore-lane prototype factors")
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

    instruments = None
    if args.instruments:
        instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    derived_spec = None
    if args.panel_dir:
        # Archived panel (explore lane): the factor column is already
        # materialized; load the day partitions directly.
        panel_dir = Path(args.panel_dir)
        if not panel_dir.is_dir():
            print(f"error: --panel-dir is not a directory: {panel_dir}")
            return 2
        try:
            dates = _resolve_dates(panel_dir, args.dates)
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
        files: list[Path] = []
        missing: list[str] = []
        for d in dates:
            flat = panel_dir / ("dt=%s.parquet" % d)
            part_dir = panel_dir / ("dt=%s" % d)
            if flat.is_file():
                files.append(flat)
            elif part_dir.is_dir():
                files.extend(sorted(part_dir.glob("*.parquet")))
            else:
                missing.append(d)
        if missing:
            print(f"error: --panel-dir partitions missing for dates: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
            return 2
        panel = pl.read_parquet([str(f) for f in files])
        if instruments:
            panel = panel.filter(pl.col("instrument").is_in(instruments))
        if panel.height == 0:
            print(f"error: no rows for {len(dates)} dates under {panel_dir}")
            return 2
        if args.factor not in panel.columns:
            print(f"error: factor column {args.factor!r} missing from the "
                  f"--panel-dir panel ({panel_dir})")
            return 2
        print(f"panel: {panel.height} rows / {len(dates)} dates from "
              f"{panel_dir} (factor {args.factor!r} pre-materialized)")
    else:
        try:
            dates = _resolve_dates(Path(cfg.out_root) / "parquet", args.dates)
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
        derived_spec = DERIVED_FACTORS.get(args.factor)
        load_factors = (
            list(derived_spec.sources) if derived_spec is not None else [args.factor]
        )
        try:
            panel = load_panel(cfg, dates, instruments=instruments, factors=load_factors)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
        if panel.height == 0:
            print(f"error: no rows for {len(dates)} dates")
            return 2
        if derived_spec is not None:
            # Admitted explore-lane factors are not materialized in the parquet;
            # recompute them from base columns exactly as the admitted spec did.
            try:
                panel = materialize_derived(panel, args.factor)
            except (KeyError, ValueError) as exc:
                print(f"error: cannot derive factor {args.factor!r}: {exc}")
                return 2
            print(
                f"derived factor {args.factor!r} materialized from panel columns "
                f"({derived_spec.doc})"
            )
        elif args.factor not in panel.columns:
            print(f"error: factor column {args.factor!r} missing from the parquet "
                  f"partitions for {len(dates)} dates")
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

    if args.matrix:
        return _run_matrix_mode(args, cfg, panel, dates, scenarios, models,
                                settle, params_yaml)

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
        base_units=args.base_units,
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
        "derived": derived_spec is not None,
        "derived_doc": derived_spec.doc if derived_spec is not None else None,
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
            "base_units": args.base_units,
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
