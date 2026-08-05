"""One-shot (eval v2 re-screen archival):
(a) dump today's explore screen reports (JSON) into one file for local archiving
(b) pairwise Spearman: each of the 6 PASS prototypes vs ALL computed prototypes
    (within-batch dedup -- the screen only dedups vs panel/engine columns)

Robust: never exits non-zero (SSH retry-loop re-run guard).
"""
import glob
import itertools
import json
import os
import time

import polars as pl

PASS6 = [
    "dd_flow_300s",
    "depth5_delta_60s",
    "flow_divergence_300s",
    "rv_asym_300s",
    "session_clock",
    "wdi_mom_90s",
]
ALL_PROTOS = [
    "depth5_delta_60s",
    "flow_divergence_300s",
    "queue_refill_asym_300s",
    "ti_ewm_state_300s",
    "ti_ewm_accel_120s",
    "iopv_premium_mom",
    "prem_x_ofi",
    "prem_x_wdi",
    "vol_adj_slope",
    "depth_resiliency",
    "session_clock",
    "ti_accum_300s",
    "ofi_accum_300s",
    "rv_asym_300s",
    "dd_flow_300s",
    "wdi_mom_90s",
    "large_trade_share_level",
    "trade_arrival_burst",
]

# ---------------- (a) today's screen reports ----------------
try:
    today_start = time.mktime(time.strptime("2026-08-05", "%Y-%m-%d"))
    latest = {}
    for p in glob.glob("/data/factor_lzt/prod/explore/reports/screen_*.json"):
        try:
            if os.path.getmtime(p) < today_start:
                continue
            stem = os.path.basename(p)[:-5]  # strip .json
            parts = stem.split("_")
            name = "_".join(parts[1:-3])
            if name not in latest or os.path.getmtime(p) > latest[name][1]:
                latest[name] = (p, os.path.getmtime(p))
        except Exception as e:
            print("skip", p, repr(e))

    reports = {}
    for name, (p, _mt) in sorted(latest.items()):
        try:
            with open(p) as f:
                payload = json.load(f)
            payload["_source_file"] = os.path.basename(p)
            reports[name] = payload
        except Exception as e:
            print("read fail", p, repr(e))

    with open(os.path.expanduser("~/rescreen_reports.json"), "w") as f:
        json.dump(reports, f, indent=1, ensure_ascii=False)
    print("reports dumped:", len(reports))
    print("names:", sorted(reports))
except Exception as e:
    print("report section failed:", repr(e))

# ---------------- (b) pairwise correlations ----------------
# explore panel layout: panels/{proto_name}/dt=YYYYMMDD.parquet
try:
    import pathlib

    base = pathlib.Path("/data/factor_lzt/prod/explore/panels")
    avail = [n for n in ALL_PROTOS if (base / n).is_dir()]
    print("proto dirs missing:", sorted(set(ALL_PROTOS) - set(avail)))
    if not avail:
        print("panels base listing:", sorted(os.listdir(base))[:8] if base.is_dir() else "MISSING")
    else:
        probe = next((base / avail[0]).glob("dt=*.parquet"))
        sch = pl.read_parquet_schema(probe)
        print("probe schema:", list(sch.keys()))
        ts_col = "ts_ms" if "ts_ms" in sch else None
        if ts_col is None:
            print("no ts_ms column; cannot align prototypes")
        else:
            per_date: dict[str, list] = {}
            for name in avail:
                for f in sorted((base / name).glob("dt=*.parquet")):
                    d = f.name.replace("dt=", "").replace(".parquet", "")
                    t = pl.read_parquet(f, columns=[ts_col, name])
                    per_date.setdefault(d, []).append(t)
            wide_frames = []
            for d in sorted(per_date):
                w = per_date[d][0]
                for t in per_date[d][1:]:
                    w = w.join(t, on=ts_col, how="inner")
                w = w.with_columns(pl.lit(d).alias("date"))
                wide_frames.append(w)
            df = pl.concat(wide_frames)
            print("wide rows:", df.height, "dates:", len(wide_frames))
            pairs = {}
            print("=== pooled Spearman: PASS6 vs all computed protos ===")
            for a in PASS6:
                if a not in avail:
                    continue
                for b in avail:
                    if b == a:
                        continue
                    sub = df.select([a, b]).drop_nulls()
                    if sub.height < 100:
                        continue
                    r = sub.select(
                        pl.corr(pl.col(a).rank(), pl.col(b).rank(), method="pearson")
                    ).item()
                    pairs[f"{a}__x__{b}"] = r
                    flag = "  <<< HIGH" if abs(r) >= 0.7 else ""
                    print(f"{a:22s} x {b:24s} n={sub.height:>7d} rho={r:+.3f}{flag}")
            with open(os.path.expanduser("~/rescreen_corr.json"), "w") as f:
                json.dump(pairs, f, indent=1)
except Exception as e:
    print("corr section failed:", repr(e))
print("done")
