"""Stage runners: factors (engine), convert (parquet), eval, mask validation.

All runners are resumable and idempotent:

* factors: per-(date, exchange, channel) jobs, skip-if-done (output exists
  and its .meta.json sidecar matches current inputs/config), sorted by input
  size descending for even disk pressure, bounded parallelism (CPU-only).
* convert: one parquet partition per day, skip-if-done against the
  partition's sidecar (an optional ``instruments`` filter is recorded there,
  so a filtered partition is never mistaken for a full one).
* eval: full Stage-4 screen + walk-forward report under reports/.  Panels
  with fewer than 5 instruments (e.g. the single-instrument pilot) skip
  cross-sectional IC -- it is undefined there -- and base every gate on the
  per-(date) time-series RankIC.
* mask: truncate-and-recompute validation per job, reports under validation/.

Every artifact goes to ``cfg.out_root`` (/data/factor_lzt) -- never into the
read-only exchange roots.
"""
from __future__ import annotations

import dataclasses
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from ..config import PipelineConfig
from ..eval.gating import GateConfig, TrialLedger, permutation_noise_floor, stage1_screen, stage2_oos_gate
from ..eval.ic import ic_stats, rank_ic_cross_section, rank_ic_time_series
from ..eval.splits import purged_day_splits
from ..ingest import DEFAULT_FACTORS, DayJob, build_day_parquet, discover_jobs, load_panel
from ..validation.golden import hash_output_csv, store_golden
from ..validation.mask_test import engine_cli_args, mask_test_day, run_engine

#: permutations used for the noise floor inside the eval stage report
_EVAL_NOISE_PERMS = 20

#: cross-sectional IC needs a real cross-section: below this many distinct
#: instruments it is undefined (rank correlation across 2-3 ETFs is noise),
#: so the eval stage skips it and bases every gate on the time-series RankIC
MIN_CROSS_SECTION_INSTRUMENTS = 5

_META_TICK_KEYS = ("tick_bytes", "ticks_bytes", "input_tick_bytes", "tick_size")
_META_SNAP_KEYS = (
    "snapshot_bytes",
    "snapshots_bytes",
    "input_snapshot_bytes",
    "snapshot_size",
)
_META_FACTOR_KEYS = ("factors", "factor_names", "factor_list")
_META_HORIZON_KEYS = ("horizons", "horizons_s")


@dataclass
class JobResult:
    job: DayJob
    status: str  # "ok" | "failed" | "skipped" | "dry_run"
    returncode: int
    elapsed_s: float
    log_tail: str


def _job_input_size(job: DayJob) -> int:
    total = 0
    for p in (job.tick_gz, job.snapshot_gz):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _meta_uptodate(job: DayJob, cfg: PipelineConfig) -> bool:
    """Skip-if-done: output exists and its sidecar matches inputs + config."""
    if not job.out_csv.is_file():
        return False
    meta_path = job.out_csv.parent / (job.out_csv.name + ".meta.json")
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    def _size_ok(keys: tuple[str, ...], path: Path) -> bool:
        for key in keys:
            if key in meta:
                try:
                    return int(meta[key]) == path.stat().st_size
                except OSError:
                    return False
        return True  # no recognizable size key -> trust the sidecar

    if not _size_ok(_META_TICK_KEYS, job.tick_gz):
        return False
    if not _size_ok(_META_SNAP_KEYS, job.snapshot_gz):
        return False

    expected_factors = list(cfg.factors) if cfg.factors else list(DEFAULT_FACTORS)
    for key in _META_FACTOR_KEYS:
        if key in meta:
            if list(meta[key]) != expected_factors:
                return False
            break
    for key in _META_HORIZON_KEYS:
        if key in meta:
            if [int(h) for h in meta[key]] != [int(h) for h in cfg.horizons_s]:
                return False
            break
    return True


def _log_path(cfg: PipelineConfig, job: DayJob) -> Path:
    return cfg.logs_dir / job.date / f"{job.exchange}_ch{job.channel}.log"


def _run_one_job(cfg: PipelineConfig, job: DayJob) -> JobResult:
    cfg.ensure_dirs()
    args = engine_cli_args(
        cfg,
        exchange=job.exchange,
        date=job.date,
        channel=job.channel,
        tick_gz=job.tick_gz,
        snapshot_gz=job.snapshot_gz,
        out_csv=job.out_csv,
    )
    started = time.monotonic()
    cp = run_engine(cfg.engine_bin, args)
    elapsed = time.monotonic() - started

    log_path = _log_path(cfg, job)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = (
        f"# hftaf-engine job {job.date} {job.exchange} ch{job.channel}\n"
        f"# args: {' '.join(str(a) for a in args)}\n"
        f"# returncode: {cp.returncode}\n"
        f"# elapsed_s: {elapsed:.2f}\n"
        f"--- stdout ---\n{cp.stdout}\n--- stderr ---\n{cp.stderr}\n"
    )
    log_path.write_text(log_text, encoding="utf-8")

    ok = cp.returncode == 0 and job.out_csv.is_file()
    return JobResult(
        job=job,
        status="ok" if ok else "failed",
        returncode=cp.returncode,
        elapsed_s=elapsed,
        log_tail=(cp.stderr or cp.stdout)[-1000:],
    )


def run_factor_stage(
    cfg: PipelineConfig,
    dates: Sequence[str],
    *,
    channels: Sequence[int] | None = None,
    max_workers: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[JobResult]:
    """Run the C++ factor/label engine over all discovered jobs."""
    jobs = discover_jobs(cfg, dates)
    if channels is not None:
        wanted = set(int(c) for c in channels)
        jobs = [j for j in jobs if j.channel in wanted]
    jobs.sort(key=_job_input_size, reverse=True)

    results: list[JobResult] = []
    runnable: list[DayJob] = []
    for job in jobs:
        if dry_run:
            results.append(
                JobResult(job=job, status="dry_run", returncode=0,
                          elapsed_s=0.0, log_tail="")
            )
        elif not overwrite and _meta_uptodate(job, cfg):
            results.append(
                JobResult(job=job, status="skipped", returncode=0,
                          elapsed_s=0.0, log_tail="up-to-date")
            )
        else:
            runnable.append(job)

    if dry_run or not runnable:
        return results

    workers = max(1, int(max_workers or cfg.max_workers))
    cfg.ensure_dirs()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one_job, cfg, job): job for job in runnable}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def run_convert_stage(
    cfg: PipelineConfig,
    dates: Sequence[str],
    *,
    overwrite: bool = False,
    instruments: Sequence[str] | None = None,
) -> list[Path]:
    """Build (or reuse) one parquet partition per date.

    ``instruments`` restricts every partition to those instrument codes; the
    filter is recorded in each partition's sidecar so a filtered partition is
    never reused where a full one is required (and vice versa).
    """
    cfg.ensure_dirs()
    paths: list[Path] = []
    for date in sorted(set(dates)):
        paths.append(
            build_day_parquet(
                date, cfg, overwrite=overwrite, instruments=instruments
            )
        )
    return paths


def _filter_dates(panel: pl.DataFrame, dates: Sequence[str]) -> pl.DataFrame:
    return panel.filter(pl.col("date").is_in(list(dates)))


def run_eval_stage(
    cfg: PipelineConfig,
    dates: Sequence[str],
    *,
    factors: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    instruments: Sequence[str] | None = None,
) -> Path:
    """Full Stage-4 evaluation: IC stats, Stage-1 screen, walk-forward gates.

    Writes ``reports/eval_{first}_{last}.json`` (+ ``.csv`` stats table) and
    returns the JSON path.  Every evaluated (factor, horizon) is appended to
    the trial ledger BEFORE thresholds are read.

    ``instruments`` restricts the panel (e.g. the 588000 single-instrument
    pilot).  Panels with fewer than :data:`MIN_CROSS_SECTION_INSTRUMENTS`
    distinct instruments skip cross-sectional IC entirely -- with one
    instrument there is no cross-section -- and ALL gating (Stage-1 screen,
    permutation floor, walk-forward OOS gate) runs on the per-(date)
    time-series RankIC aggregated with Newey-West t.
    """
    cfg.ensure_dirs()
    dates_sorted = sorted(set(dates))
    panel = load_panel(cfg, dates_sorted, factors=factors, instruments=instruments)

    factor_cols = (
        [f for f in factors if f in panel.columns]
        if factors
        else [f for f in DEFAULT_FACTORS if f in panel.columns]
    )
    if not factor_cols:
        raise ValueError("no factor columns available for evaluation")
    horizons_eff = [int(h) for h in (horizons or cfg.horizons_s)]

    n_instruments = panel["instrument"].n_unique() if panel.height else 0
    cross_enabled = n_instruments >= MIN_CROSS_SECTION_INSTRUMENTS

    ledger = TrialLedger(cfg.reports_dir / "trial_ledger.jsonl")
    gate_cfg = GateConfig()

    stats_list = []
    cross_list = []
    noise_floors: dict[tuple[str, int], float] = {}
    for f in factor_cols:
        for h in horizons_eff:
            ic_ts = rank_ic_time_series(panel, f, h)
            stats_list.append(ic_stats(ic_ts, f, h, max_lag=h // 3))
            if cross_enabled:
                ic_xs = rank_ic_cross_section(panel, f, h)
                cross_list.append(ic_stats(ic_xs, f, h, ic_col="ic"))
            noise_floors[(f, h)] = permutation_noise_floor(
                panel, f, h, n_perms=_EVAL_NOISE_PERMS
            )

    screen_df = stage1_screen(stats_list, ledger, gate_cfg, noise_floors)

    # ------- day-blocked purged walk-forward (IS/OOS per fold) ----------
    splits = purged_day_splits(dates_sorted)
    walk_forward: list[dict] = []
    for f, h in [(s.factor, s.horizon_s) for s in stats_list]:
        for i, split in enumerate(splits):
            is_panel = _filter_dates(panel, split.train_dates)
            oos_panel = _filter_dates(panel, split.test_dates)
            if is_panel.is_empty() or oos_panel.is_empty():
                continue
            is_st = ic_stats(rank_ic_time_series(is_panel, f, h), f, h, max_lag=h // 3)
            oos_st = ic_stats(rank_ic_time_series(oos_panel, f, h), f, h, max_lag=h // 3)
            # honest-N bookkeeping: every walk-forward evaluation is a trial
            ledger.log(
                factor=f,
                horizon_s=h,
                params={"fold": i},
                stage="stage2_walkforward",
                metrics={
                    "is_mean_ic": is_st.mean_ic,
                    "oos_mean_ic": oos_st.mean_ic,
                    "oos_t_stat_nw": oos_st.t_stat_nw,
                },
            )
            passed, details = stage2_oos_gate(is_st, oos_st, gate_cfg)
            walk_forward.append(
                {
                    "factor": f,
                    "horizon_s": h,
                    "fold": i,
                    "train_dates": list(split.train_dates),
                    "test_dates": list(split.test_dates),
                    "is_mean_ic": is_st.mean_ic,
                    "oos_mean_ic": oos_st.mean_ic,
                    "oos_t_stat_nw": oos_st.t_stat_nw,
                    "passed": passed,
                    "details": details,
                }
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dates": dates_sorted,
        "factors": factor_cols,
        "horizons_s": horizons_eff,
        "instruments": sorted(panel["instrument"].unique().to_list()),
        "n_instruments": n_instruments,
        "cross_section_skipped": not cross_enabled,
        "stats": [dataclasses.asdict(s) for s in stats_list],
        "cross_section_stats": [dataclasses.asdict(s) for s in cross_list],
        "noise_floors": [
            {"factor": f, "horizon_s": h, "floor": v}
            for (f, h), v in sorted(noise_floors.items())
        ],
        "stage1_screen": screen_df.to_dicts(),
        "walk_forward": walk_forward,
        "n_trials_total": ledger.n_trials(),
    }

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and obj != obj:  # NaN -> null in JSON
            return None
        return obj

    stamp = f"{dates_sorted[0]}_{dates_sorted[-1]}"
    report_path = cfg.reports_dir / f"eval_{stamp}.json"
    report_path.write_text(
        json.dumps(_sanitize(report), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    stats_df = pl.DataFrame([dataclasses.asdict(s) for s in stats_list])
    stats_df.write_csv(report_path.with_suffix(".csv"))
    return report_path


def run_mask_stage(
    cfg: PipelineConfig, dates: Sequence[str], *, k: int = 4
) -> Path:
    """Run the lookahead mask test for every discovered job with data.

    Jobs whose raw output does not exist yet are produced by the engine as
    part of the test.  Passed jobs also store a golden hash under
    ``validation/golden`` for future regression checks.
    """
    cfg.ensure_dirs()
    jobs = discover_jobs(cfg, dates)
    entries: list[dict] = []
    for job in jobs:
        entry: dict = {
            "date": job.date,
            "exchange": job.exchange,
            "channel": job.channel,
        }
        try:
            rep = mask_test_day(cfg, job.date, job.exchange, job.channel, k=k)
            entry["report"] = dataclasses.asdict(rep)
            entry["error"] = None
            if rep.passed:
                store_golden(
                    job.date, job.exchange, job.channel,
                    hash_output_csv(job.out_csv), cfg,
                )
        except Exception as exc:  # keep going; record and fail the stage
            entry["report"] = None
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entries.append(entry)

    dates_sorted = sorted(set(dates))
    stamp = f"{dates_sorted[0]}_{dates_sorted[-1]}" if dates_sorted else "none"
    report_path = cfg.validation_dir / f"mask_report_{stamp}.json"
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": k,
        "n_jobs": len(entries),
        "n_passed": sum(
            1 for e in entries if e["report"] and e["report"]["passed"]
        ),
        "entries": entries,
    }
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return report_path
