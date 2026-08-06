"""Deterministic round runner for the explore iteration lane.

``hftaf-explore round`` runs the whole post-research pipeline for one iter
round in a single command:

    add specs -> trading-day list -> parallel run (chunked) -> screen ->
    extract -> pooled-Spearman dedup -> archive (lossless candidates.json
    merge) -> result bundle.

Idempotent: run/screen skip when their outputs already exist
(``run_prototype`` is skip-if-done; later stages skip on existing JSON), so
a half-finished round resumes cheaply. No git: the LLM reviews the bundle,
fills the round's prose (trigger / lessons / death notes) in candidates.json,
re-runs ``--stage render``, and commits as DracoMeowa.

Folds the per-round scratch scripts (round{N}_add / extract_round{N} /
round{N}_corr / build_round{N}_archive / render_round_report) into
parameterized functions so they stop being cloned every round.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from .layout import (
    panel_path,
    panels_dir,
    reports_dir,
    spec_meta_path,
    spec_path,
)

GATE_CRITERIA = (
    "RankIC-primary: per horizon IS+OOS Newey-West |t|>=2.0, retention>=0.5 "
    "same sign; panel |Spearman|<=0.85; pooled dedup |Spearman|<=0.85."
)
DEDUP_RULE = (
    "In |rho|>=0.85 pairs keep higher max|OOS NW t|; tiebreak n_passed_horizons "
    "then panel orthogonality. rho>=0.999 = identical series -> library "
    "precedence unless >20% stronger."
)
SPLIT_TEXT = "train 60d / OOS 5d + 1d embargo, purged day-blocked walk-forward"
HORIZONS_S = [15, 30, 60, 300, 900]
COST_PER_SIDE_BPS = 3


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def trading_days(cfg) -> list[str]:
    """Explicit trading days = every parquet partition actually present.

    Replaces the fragile ``A..B`` date spec (``expand_date_spec`` yields
    calendar days incl. weekends, which choke on missing Saturday partitions).
    """
    pdir = cfg.parquet_dir
    days = sorted(
        p.name.split("=", 1)[1]
        for p in pdir.glob("dt=*")
        if (p / "factors.parquet").is_file()
    )
    if not days:
        raise FileNotFoundError(f"no parquet partitions under {pdir}")
    return days


def iterations_dir(cfg, round_key: str) -> Path:
    d = Path(cfg.out_root) / "iterations" / round_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def library_path(repo_root: Path) -> Path:
    return repo_root / "library" / "candidates.json"


def scratch_dir(repo_root: Path) -> Path:
    return repo_root / "scratch-iter003"


def _explore_cmd() -> list[str]:
    exe = shutil.which("hftaf-explore")
    return [exe] if exe else [sys.executable, "-m", "hft_autofactor.explore"]


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def load_round_manifests(
    repo_root: Path, round_key: str
) -> tuple[list[str], dict[str, str], dict[str, int]]:
    """Return (names, name->family letter, letter->count)."""
    n = int(round_key.split("_round")[-1])
    scratch = scratch_dir(repo_root)
    names: list[str] = []
    fam_of: dict[str, str] = {}
    families: dict[str, int] = {}
    for mf in sorted(scratch.glob(f"manifest_iter003_R{n}*.json")):
        letter = mf.stem.split("_")[-1]  # e.g. R6A
        entries = json.loads(mf.read_text(encoding="utf-8"))
        families[letter] = len(entries)
        for e in entries:
            names.append(e["name"])
            fam_of[e["name"]] = letter
    if not names:
        raise FileNotFoundError(
            f"no manifests manifest_iter003_R{n}*.json under {scratch}"
        )
    return names, fam_of, families


# --------------------------------------------------------------------------- #
# Stage: add
# --------------------------------------------------------------------------- #
def add_specs(cfg, spec_dir: Path, names: list[str]) -> dict:
    from .registry import PrototypeError, load_prototype_spec

    added = skipped = failed = 0
    for n in names:
        src = spec_dir / f"{n}.py"
        if not src.is_file():
            print(f"  add MISSING {src}", file=sys.stderr)
            failed += 1
            continue
        dst = spec_path(cfg, n)
        if dst.is_file():
            skipped += 1
            continue
        try:
            proto = load_prototype_spec(src, source=str(src))
        except PrototypeError as exc:
            print(f"  add FAIL {n}: {exc}", file=sys.stderr)
            failed += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        spec_meta_path(cfg, n).write_text(
            json.dumps(proto.metadata_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        added += 1
    return {"added": added, "skipped": skipped, "failed": failed}


# --------------------------------------------------------------------------- #
# Stage: parallel run (chunked subprocesses)
# --------------------------------------------------------------------------- #
def _chunks(seq: list, n: int) -> list:
    n = max(1, n)
    return [seq[i::n] for i in range(n)]


def parallel_run(
    cfg,
    config_path: str,
    names: list[str],
    dates: list[str],
    *,
    k: int = 8,
    chunk_days: int = 5,
    workers: int = 4,
) -> dict:
    """Chunk protos into ``workers`` disjoint groups and run them concurrently.

    Each proto writes only its own ``panels/{name}/`` (disjoint paths), reads
    the shared wide table read-only -> safe parallelism, no engine change.
    """
    missing = [
        n for n in names
        if not all(panel_path(cfg, n, d).is_file() for d in dates)
    ]
    if not missing:
        print("  run: all panels up-to-date, skipping")
        return {"status": "skipped", "ran": 0, "workers": workers}

    dates_csv = ",".join(dates)
    chunks = [c for c in _chunks(missing, workers) if c]

    def run_chunk(idx: int, chunk: list[str]):
        proc = subprocess.run(
            _explore_cmd() + [
                "--config", config_path, "run",
                "--dates", dates_csv, "--protos", ",".join(chunk),
                "--k", str(k), "--chunk-days", str(chunk_days),
            ],
            capture_output=True, text=True,
        )
        return idx, proc.returncode, proc.stdout, proc.stderr

    print(f"  run: {len(missing)} proto(s) in {len(chunks)} chunk(s), workers={workers}")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_chunk, i, c) for i, c in enumerate(chunks)]
        for f in futs:
            results.append(f.result())
    bad = 0
    for idx, rc, out, err in results:
        last = (out or "").strip().splitlines()
        tail = last[-1] if last else "(no output)"
        print(f"    chunk {idx}: rc={rc} | {tail}")
        if rc not in (0, 1) and err.strip():
            print(f"      stderr: {err.strip().splitlines()[-1][:200]}", file=sys.stderr)
            bad += 1
    return {"status": "ran", "ran": len(missing), "workers": workers, "chunk_bad": bad}


# --------------------------------------------------------------------------- #
# Stage: screen
# --------------------------------------------------------------------------- #
def _latest_screen(cfg, name: str, dates: list[str]):
    if not dates:
        return None
    fs = sorted(
        reports_dir(cfg).glob(f"screen_{name}_{dates[0]}_{dates[-1]}_*.json")
    )
    return fs[-1] if fs else None


def screen(cfg, config_path: str, names: list[str], dates: list[str]) -> int:
    if all(_latest_screen(cfg, n, dates) for n in names):
        print("  screen: all reports present, skipping")
        return 0
    proc = subprocess.run(
        _explore_cmd() + [
            "--config", config_path, "screen",
            "--dates", ",".join(dates), "--protos", ",".join(names),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):
        print(proc.stdout + proc.stderr, file=sys.stderr)
        raise RuntimeError(f"screen failed rc={proc.returncode}")
    return proc.returncode


# --------------------------------------------------------------------------- #
# Stage: extract (port of extract_roundN.py)
# --------------------------------------------------------------------------- #
def extract_all(cfg, round_key, names, fam_of, dates) -> dict:
    out_json = iterations_dir(cfg, round_key) / "all.json"
    if out_json.is_file():
        print(f"  extract: {out_json} exists, skipping")
        return json.loads(out_json.read_text(encoding="utf-8"))

    out = {"pass": [], "fail": []}
    for name in names:
        f = _latest_screen(cfg, name, dates)
        if f is None:
            out["fail"].append(
                {"name": name, "family": fam_of.get(name, "?"), "status": "no_report"}
            )
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        dc = r.get("duplicate_check", {}) or {}
        entry = {
            "name": name,
            "family": fam_of.get(name, "?"),
            "status": r.get("status"),
            "passed": r.get("passed"),
            "reasons": r.get("reasons"),
            "mechanism": (r.get("prototype") or {}).get("mechanism", ""),
            "panel_max_corr": dc.get("max_abs_corr"),
            "panel_corr_factor": dc.get("library_factor"),
            "horizons": {},
        }
        for h in r.get("horizons", []):
            entry["horizons"][str(h["horizon_s"])] = {
                "passed": h.get("passed"),
                "is_mean_ic": h.get("is_mean_ic"),
                "is_t": h.get("is_t_stat_nw"),
                "oos_mean_ic": h.get("oos_mean_ic"),
                "oos_t": h.get("oos_t_stat_nw"),
                "retention": h.get("retention"),
                "head_stats": h.get("head_stats"),
            }
        (out["pass"] if r.get("passed") else out["fail"]).append(entry)

    out_json.write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  extract: pass={len(out['pass'])} fail={len(out['fail'])} -> {out_json}")
    return out


# --------------------------------------------------------------------------- #
# Stage: pooled-Spearman dedup (port of roundN_corr.py)
# --------------------------------------------------------------------------- #
def build_library_names(lib: dict) -> list[str]:
    names = []
    for k, v in lib.items():
        if k.startswith("iter003_round"):
            for a in v.get("admitted", []):
                names.append(a["name"])
    for a in lib.get("eval_v2_rescreen_2026_08_05", {}).get("verdicts", []):
        if a.get("passed_horizons"):
            names.append(a["name"])
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _load_panel_pooled(cfg, name: str) -> pl.DataFrame:
    d = panels_dir(cfg, name)
    files = sorted(d.glob("dt=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no panels for {name} in {d}")
    parts = []
    for f in files:
        df = pl.read_parquet(f)
        if name not in df.columns:
            raise KeyError(f"column {name} missing in {f}: {df.columns}")
        if "date" not in df.columns:
            df = df.with_columns(pl.lit(f.stem.split("=")[-1]).alias("date"))
        parts.append(
            df.select(["date", "ts_ms", name]).with_columns(
                pl.col("date").cast(pl.Utf8), pl.col("ts_ms").cast(pl.Int64)
            )
        )
    return pl.concat(parts, how="vertical")


def _spearman(x: pl.Series, y: pl.Series) -> float:
    rx, ry = x.rank(), y.rank()
    dx, dy = rx - rx.mean(), ry - ry.mean()
    num = float((dx * dy).sum())
    den = (float((dx * dx).sum()) * float((dy * dy).sum())) ** 0.5
    return float("nan") if den <= 0 else num / den


def corr_dedup(cfg, round_key, pass_names, repo_root) -> dict:
    out_json = iterations_dir(cfg, round_key) / "corr.json"
    out_txt = iterations_dir(cfg, round_key) / "corr.txt"
    if out_json.is_file():
        print(f"  corr: {out_json} exists, skipping")
        return json.loads(out_json.read_text(encoding="utf-8"))

    if not pass_names:
        payload = {"n_rows_pooled": 0, "n_dates": 0, "n_library": 0, "names": [], "pairs": []}
        out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return payload

    lib = json.loads(library_path(repo_root).read_text(encoding="utf-8"))
    library = build_library_names(lib)
    names = pass_names + library
    print(f"  corr: library={len(library)} pass_new={len(pass_names)} total={len(names)}")

    wide = None
    for n in names:
        df = _load_panel_pooled(cfg, n)
        wide = df if wide is None else wide.join(df, on=["date", "ts_ms"], how="inner")
    wide = wide.sort(["date", "ts_ms"])

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sub = wide.select([a, b]).drop_nulls()
            n_obs = sub.height
            rho = _spearman(sub[a], sub[b]) if n_obs >= 1000 else float("nan")
            pairs.append({"a": a, "b": b, "rho": rho, "n": n_obs})
    pairs.sort(key=lambda p: -abs(p["rho"]) if p["rho"] == p["rho"] else 0)

    payload = {
        "n_rows_pooled": wide.height,
        "n_dates": int(wide.select("date").n_unique()),
        "n_library": len(library),
        "names": names,
        "pairs": pairs,
    }
    out_json.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        f"pooled rows={wide.height} dates={wide.select('date').n_unique()} "
        f"library={len(library)}"
    ]
    for p in pairs:
        if p["rho"] != p["rho"] or abs(p["rho"]) < 0.50:
            continue
        tag = "  <<< HIGH" if abs(p["rho"]) >= 0.85 else ""
        lines.append(
            f"{p['a']:36s} x {p['b']:36s} n={p['n']:7d} rho={p['rho']:+.3f}{tag}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_high = sum(1 for p in pairs if p["rho"] == p["rho"] and abs(p["rho"]) >= 0.85)
    print(f"  corr: {len(pairs)} pairs, {n_high} HIGH -> {out_json}")
    return payload


# --------------------------------------------------------------------------- #
# Stage: archive (port of build_roundN_archive.py, mechanical parts only)
# --------------------------------------------------------------------------- #
def _best_t(entry):
    if not entry:
        return (0.0, 0)
    ph = {h: d for h, d in entry.get("horizons", {}).items() if d.get("passed")}
    if not ph:
        return (0.0, 0)
    h, d = max(ph.items(), key=lambda kv: abs(kv[1].get("oos_t") or 0))
    return (d.get("oos_t") or 0.0, len(ph))


def _lib_entry(name: str, lib: dict):
    for k, v in lib.items():
        if k.startswith("iter003_round"):
            for a in v.get("admitted", []):
                if a["name"] == name:
                    return a
    for a in lib.get("eval_v2_rescreen_2026_08_05", {}).get("verdicts", []):
        if a.get("name") == name:
            return {"name": name, "horizons": a.get("stats", {})}
    return None


def _classify_fail(e: dict) -> str:
    hz = e.get("horizons", {})
    if e.get("status") == "rejected_duplicate":
        return "panel_duplicate"
    has_is = any(abs(h.get("is_t") or 0) >= 2.0 for h in hz.values())
    if not has_is:
        return "is_dead"
    if any(
        abs(h.get("is_t") or 0) >= 2.0
        and abs(h.get("oos_t") or 0) >= 2.0
        and (h.get("retention") or 0) < 0.5
        for h in hz.values()
    ):
        return "retention_or_sign"
    if any(
        abs(h.get("is_t") or 0) >= 2.0 and abs(h.get("oos_t") or 0) < 2.0
        for h in hz.values()
    ):
        return "oos_collapse"
    return "is_dead"


def build_archive(cfg, repo_root, round_key, allr, corr, fam_of, families) -> dict:
    libp = library_path(repo_root)
    lib = json.loads(libp.read_text(encoding="utf-8"))
    passd = {e["name"]: e for e in allr["pass"]}
    faild = allr["fail"]
    n_round = int(round_key.split("_round")[-1])
    old = lib.get(round_key, {})  # preserve prose across re-runs

    def is_lib(name: str) -> bool:
        return name not in passd and _lib_entry(name, lib) is not None

    # ---- dedup over HIGH pairs (|rho|>=0.85) ----
    high = [p for p in corr["pairs"] if abs(p["rho"]) >= 0.85]
    killed = []
    for p in high:
        a, b, rho = p["a"], p["b"], p["rho"]
        ea = passd.get(a) or _lib_entry(a, lib)
        eb = passd.get(b) or _lib_entry(b, lib)
        ta, tb = _best_t(ea), _best_t(eb)
        if rho >= 0.999:
            if is_lib(a) and not (abs(tb[0]) > abs(ta[0]) * 1.2):
                keep, kill = a, b
            elif is_lib(b) and not (abs(ta[0]) > abs(tb[0]) * 1.2):
                keep, kill = b, a
            else:
                keep, kill = (a, b) if abs(ta[0]) >= abs(tb[0]) else (b, a)
            rule = ("rho>=0.999 near-identical series; library precedence "
                    "unless >20% stronger")
        else:
            keep, kill = (
                (a, b)
                if (abs(ta[0]) > abs(tb[0])
                    or (abs(ta[0]) == abs(tb[0]) and ta[1] >= tb[1]))
                else (b, a)
            )
            rule = "keep higher |OOS t| (rho in [0.85,0.999))"
        kt = ta if kill == a else tb
        killed.append({
            "name": kill, "against": keep, "rho": rho,
            "kill_oos_t": round(kt[0], 2), "kill_n_h": kt[1],
            "killed_reason": f"{rule}; kill |OOS t|={kt[0]:.2f} ({kt[1]}h)",
        })

    killed_names = {k["name"] for k in killed}
    admitted_names = [
        e["name"] for e in allr["pass"] if e["name"] not in killed_names
    ]
    admitted = []
    for name in admitted_names:
        e = passd[name]
        bo_h, _ = max(
            ((h, d) for h, d in e["horizons"].items() if d["passed"]),
            key=lambda hd: abs(hd[1]["oos_t"]),
        )
        pmc = e.get("panel_max_corr")
        admitted.append({
            "name": name,
            "family": fam_of.get(name, "?"),
            "mechanism": e["mechanism"],
            "max_abs_corr_vs_panel": round(pmc, 3) if pmc is not None else None,
            "panel_corr_factor": e.get("panel_corr_factor"),
            "best_horizon_s": bo_h,
            "horizons": e["horizons"],
        })
    admitted.sort(
        key=lambda a: -abs(a["horizons"][a["best_horizon_s"]]["oos_t"])
    )

    # ---- death map (mechanical; notes left for the LLM) ----
    by_fm: dict = {}
    for e in faild:
        key = (fam_of.get(e["name"], "?"), _classify_fail(e))
        by_fm.setdefault(key, []).append(e["name"])
    death_map = {
        "by_family_dominant_mode": {
            f"{f}/{m}": ns for (f, m), ns in sorted(by_fm.items())
        },
        "notes": old.get("death_map", {}).get("notes") or [],
    }

    # ---- library totals (exclude the block being built) ----
    lib_names = set()
    for k, v in lib.items():
        if k.startswith("iter003_round") and k != round_key:
            for a in v.get("admitted", []):
                lib_names.add(a["name"])
    for a in lib.get("eval_v2_rescreen_2026_08_05", {}).get("verdicts", []):
        if a.get("passed_horizons"):
            lib_names.add(a["name"])
    surv = set(admitted_names) | lib_names
    watch = [
        f"{p['a']} x {p['b']} rho={p['rho']:+.3f}"
        for p in corr["pairs"]
        if 0.70 <= abs(p["rho"]) < 0.85 and p["a"] in surv and p["b"] in surv
    ]
    lib_total = len(lib_names) + len(admitted_names)

    block = {
        "trigger": old.get("trigger") or "",
        "generated_at": time.strftime("%Y-%m-%d"),
        "batch": {"n_specs": len(fam_of), "families": families},
        "gate": {
            "criteria": GATE_CRITERIA,
            "horizons_s": HORIZONS_S,
            "cost_per_side_bps": COST_PER_SIDE_BPS,
            "head_long_short": "descriptive only (tau 1/5/10%)",
        },
        "split": SPLIT_TEXT,
        "pipeline_run": old.get("pipeline_run") or "",
        "summary": {
            "screen_pass": len(allr["pass"]),
            "screen_fail": len(allr["fail"]),
            "dedup_killed": len(killed),
            "admitted_new": len(admitted),
            "library_total": lib_total,
        },
        "dedup": {"rule": DEDUP_RULE, "killed": killed, "watchlist_subthreshold": watch},
        "admitted": admitted,
        "death_map": death_map,
        "lessons": old.get("lessons") or [],
        "artifacts": {
            "candidates_key": round_key,
            "iterations_dir": str(iterations_dir(cfg, round_key)),
            "detail": f"scratch-iter003/round{n_round}_admitted_detail.json, "
                      f"round{n_round}_deathmap.json",
            "corr": f"iterations/{round_key}/corr.json/.txt",
        },
    }

    sc = scratch_dir(repo_root)
    (sc / f"round{n_round}_admitted_detail.json").write_text(
        json.dumps(admitted, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (sc / f"round{n_round}_deathmap.json").write_text(
        json.dumps(death_map, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    # ---- lossless merge into candidates.json ----
    before = json.dumps(
        {k: v for k, v in lib.items() if k != round_key},
        sort_keys=True, ensure_ascii=False,
    )
    lib[round_key] = block
    after = json.dumps(
        {k: v for k, v in lib.items() if k != round_key},
        sort_keys=True, ensure_ascii=False,
    )
    assert before == after, "merge not lossless!"
    libp.write_text(
        json.dumps(lib, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"  archive: admitted={len(admitted)} killed={len(killed)} "
        f"library_total={lib_total} watch={len(watch)}"
    )
    return block


# --------------------------------------------------------------------------- #
# Bundle + family table
# --------------------------------------------------------------------------- #
def _family_table(block, allr, fam_of) -> dict:
    pass_per: dict = {}
    for e in allr["pass"]:
        f = e.get("family") or fam_of.get(e["name"], "?")
        pass_per[f] = pass_per.get(f, 0) + 1
    admit_per: dict = {}
    for a in block["admitted"]:
        f = a.get("family") or fam_of.get(a["name"], "?")
        admit_per[f] = admit_per.get(f, 0) + 1
    return {
        f: {
            "submitted": block["batch"]["families"].get(f, "?"),
            "pass": pass_per.get(f, 0),
            "admitted": admit_per.get(f, 0),
        }
        for f in block["batch"]["families"]
    }


def emit_bundle(cfg, round_key, block, allr, corr, fam_of) -> dict:
    admitted = block["admitted"]
    top = sorted(
        admitted,
        key=lambda a: -abs(a["horizons"][a["best_horizon_s"]]["oos_t"]),
    )[:8]
    dead_modes: dict = {}
    for key, ns in block["death_map"]["by_family_dominant_mode"].items():
        mode = key.split("/", 1)[1] if "/" in key else key
        dead_modes[mode] = dead_modes.get(mode, 0) + len(ns)

    bundle = {
        "round": round_key,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": block["summary"],
        "family_table": _family_table(block, allr, fam_of),
        "admitted_top": [
            {
                "name": a["name"],
                "family": a["family"],
                "best_horizon_s": a["best_horizon_s"],
                "oos_t": a["horizons"][a["best_horizon_s"]]["oos_t"],
                "oos_mean_ic": a["horizons"][a["best_horizon_s"]]["oos_mean_ic"],
                "n_passed": sum(1 for h in a["horizons"].values() if h.get("passed")),
                "mechanism": a["mechanism"],
            }
            for a in top
        ],
        "dedup_killed": block["dedup"]["killed"],
        "death_modes": dead_modes,
        "death_by_family": block["death_map"]["by_family_dominant_mode"],
        "fail_detail_count": len(allr["fail"]),
        "n_rows_pooled": corr.get("n_rows_pooled", 0),
        "next_round_brief_seed": {
            "strongest_clusters": [a["name"] for a in top[:4]],
            "dead_dominant_modes": sorted(dead_modes, key=lambda k: -dead_modes[k])[:3],
            "watchlist_subthreshold": block["dedup"]["watchlist_subthreshold"][:10],
        },
        "prose_to_fill": ["trigger", "pipeline_run", "death_map.notes", "lessons"],
        "next_step": (
            f"fill prose in candidates.json[{round_key!r}] "
            f"(trigger/pipeline_run/death_map.notes/lessons), then run "
            f"`hftaf-explore round --stage render --round {round_key}`, "
            f"then commit as DracoMeowa"
        ),
    }
    out = iterations_dir(cfg, round_key) / "bundle.json"
    out.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  bundle -> {out}")
    return bundle


# --------------------------------------------------------------------------- #
# Stage: render (port of render_round_report.py)
# --------------------------------------------------------------------------- #
def _best_oos(horizons: dict) -> dict:
    passed = {h: d for h, d in horizons.items() if d.get("passed")}
    if not passed:
        return {}
    h, d = max(passed.items(), key=lambda kv: abs(kv[1]["oos_t"]))
    return {"h": h, "oos_ic": d["oos_mean_ic"], "oos_t": d["oos_t"], "n_passed": len(passed)}


def render_report(repo_root: Path, round_key: str) -> Path:
    lib = json.loads(library_path(repo_root).read_text(encoding="utf-8"))
    b = lib[round_key]
    s = b["summary"]
    n_round = round_key.split("_round")[-1]
    _, fam_of, _ = load_round_manifests(repo_root, round_key)

    fam_admit: dict = {L: 0 for L in b["batch"]["families"]}
    fam_killed: dict = {L: set() for L in b["batch"]["families"]}
    for e in b["admitted"]:
        L = fam_of.get(e["name"])
        if L in fam_admit:
            fam_admit[L] += 1
    for k in b["dedup"]["killed"]:
        L = fam_of.get(k["name"])
        if L in fam_killed:
            fam_killed[L].add(k["name"])
    fam_pass: dict = {
        L: fam_admit[L] + len(fam_killed[L]) for L in b["batch"]["families"]
    }

    out = []
    out.append(f"# iter-003 第{n_round}轮评估报告（588000，59 列宽表持续迭代）\n")
    out.append(f"日期：{b['generated_at']}　批次：{round_key}")
    out.append(
        f"前置：库 {s['library_total'] - s['admitted_new']}；"
        f"本轮按经验定向挖 {len(b['batch']['families'])} 个方向。\n"
    )
    out.append("## 一句话结果\n")
    out.append(
        f"{b['batch']['n_specs']} 个候选 → eval-v2 门槛过 {s['screen_pass']} → "
        f"对 {s['library_total'] - s['admitted_new']} 因子库 pooled Spearman "
        f"去重砍 {s['dedup_killed']} → "
        f"**入库 {s['admitted_new']}，库总计 {s['library_total']}**。\n"
    )

    out.append("## 批次构成\n")
    out.append("| 组 | 个数 | 过筛 | 入库 |")
    out.append("|---|---|---|---|")
    for L, cnt in b["batch"]["families"].items():
        out.append(f"| {L} | {cnt} | {fam_pass.get(L, '?')} | {fam_admit.get(L, '?')} |")
    out.append("")

    out.append("## 入库因子（按最强 OOS |t| 排序）\n")
    rows = [(e, _best_oos(e["horizons"])) for e in b["admitted"]]
    rows.sort(key=lambda r: -abs(r[1].get("oos_t", 0)))
    out.append("| 因子 | 含义 | 过几个h | 最强 OOS IC（t） | 面板\\|ρ\\| |")
    out.append("|---|---|---|---|---|")
    for e, bo in rows:
        if not bo:
            continue
        mech = e["mechanism"][:40] + ("…" if len(e["mechanism"]) > 40 else "")
        out.append(
            f"| {e['name']} | {mech} | {bo.get('n_passed', '?')} | "
            f"{bo['h']}s {bo['oos_ic']:+.4f}（{bo['oos_t']:+.2f}） | "
            f"{e.get('max_abs_corr_vs_panel')} |"
        )
    out.append("")

    out.append("## 去重\n")
    if b["dedup"]["killed"]:
        out.append(f"砍 {len(b['dedup']['killed'])} 个：\n")
        for k in b["dedup"]["killed"]:
            out.append(f"- **{k['name']}**：{k['killed_reason']}")
        out.append("")
    if b["dedup"].get("watchlist_subthreshold"):
        out.append("观察名单（0.70–0.85，记账不砍）：")
        for w in b["dedup"]["watchlist_subthreshold"]:
            out.append(f"- {w}")
        out.append("")

    out.append("## 死因地图\n")
    dm = b["death_map"]
    if dm.get("by_family_dominant_mode"):
        out.append("| 组/模式 | 个数 | 因子 |")
        out.append("|---|---|---|")
        for key, names in sorted(dm["by_family_dominant_mode"].items()):
            out.append(f"| {key} | {len(names)} | {', '.join(names)} |")
        out.append("")
    for note in dm.get("notes", []):
        out.append(f"- {note}")
    out.append("")

    out.append("## 经验沉淀\n")
    for L in b.get("lessons", []):
        out.append(f"- {L}")
    out.append("")

    out.append("## 归档\n")
    out.append(
        f"- library/candidates.json 键 `{round_key}`；"
        f"scratch-iter003/：round{n_round}_admitted_detail.json、"
        f"round{n_round}_deathmap.json、本报告。"
        f"服务器副本 /data/factor_lzt/prod/iterations/{round_key}/。"
    )

    md = "\n".join(out) + "\n"
    target = scratch_dir(repo_root) / f"{round_key}_report.md"
    target.write_text(md, encoding="utf-8")
    print(f"  render: {target} ({len(md)} chars)")
    return target


# --------------------------------------------------------------------------- #
# Top-level orchestrator
# --------------------------------------------------------------------------- #
def run_round(
    config_path: str,
    repo_root: str | Path,
    round_key: str,
    spec_dir: str | Path,
    *,
    workers: int = 4,
    k: int = 8,
    chunk_days: int = 5,
    stage: str = "all",
) -> dict:
    """Run one iter round through the deterministic post-research pipeline.

    stage: ``all`` = add+run+screen+extract+corr+archive+bundle (no render);
           ``run`` = add+run+screen only; ``archive`` = extract+corr+archive+bundle;
           ``render`` = render report from candidates.json (after LLM fills prose).
    """
    from ..config import load_config

    cfg = load_config(config_path)
    repo_root = Path(repo_root).resolve()
    spec_dir = Path(spec_dir).resolve()
    print(f"=== round {round_key} stage={stage} workers={workers} k={k} ===")

    names, fam_of, families = load_round_manifests(repo_root, round_key)
    print(f"manifests: {len(names)} protos, families={families}")
    dates = trading_days(cfg)
    print(f"trading days: {len(dates)} ({dates[0]}..{dates[-1]})")

    if stage in ("all", "run"):
        print("-- add --")
        print(f"add: {add_specs(cfg, spec_dir, names)}")
        print("-- run --")
        print(f"run: {parallel_run(cfg, config_path, names, dates, k=k, chunk_days=chunk_days, workers=workers)}")
        print("-- screen --")
        screen(cfg, config_path, names, dates)
        print("screen: done")

    bundle = None
    if stage in ("all", "archive"):
        print("-- extract --")
        allr = extract_all(cfg, round_key, names, fam_of, dates)
        pass_names = [e["name"] for e in allr["pass"]]
        print("-- corr --")
        corr = corr_dedup(cfg, round_key, pass_names, repo_root)
        print("-- archive --")
        block = build_archive(cfg, repo_root, round_key, allr, corr, fam_of, families)
        print("-- bundle --")
        bundle = emit_bundle(cfg, round_key, block, allr, corr, fam_of)
        print("\n=== BUNDLE (summary) ===")
        print(json.dumps(
            {
                "round": bundle["round"],
                "summary": bundle["summary"],
                "family_table": bundle["family_table"],
                "death_modes": bundle["death_modes"],
                "admitted_top_count": len(bundle["admitted_top"]),
                "next_step": bundle["next_step"],
            },
            ensure_ascii=False, indent=2,
        ))

    if stage == "render":
        render_report(repo_root, round_key)

    return bundle or {"stage": stage, "ok": True}
