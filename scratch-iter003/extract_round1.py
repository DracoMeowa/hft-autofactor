"""Build iter-003 round-1 archive data: admitted detail JSON + fail table.

Reads local screen report copies (reports_round1/) and emits:
  - round1_admitted_detail.json : per-admitted proto, per-passed-horizon
    IC/t/retention + head stats (for candidates.json block insertion)
  - prints a compact FAIL table (best IS/OOS stats) for the report narrative.
"""
import json
from pathlib import Path

HERE = Path(r"D:\claude\Quant_works\hft-autofactor\scratch-iter003")
REP = HERE / "reports_round1"

ADMITTED = [
    "depth5_delta_120s", "depth5_delta_30s", "flow_divergence_120s",
    "flow_divergence_60s", "flow_divergence_x_spread_z", "last_mid_gap_ticks",
    "log_mid_ret_120s", "mid_day_range_pos", "ofi_z_x_spread_z", "oir_mom_60s",
    "price_accel_60_180", "signed_rv_60s", "top_book_delta_120s",
    "wdi_accel_90s", "wdi_mom_180s", "wdi_mom_30s",
]
DEDUP_KILLED = ["microprice_dev_mom_60s", "vol_adj_mom_60s", "top_book_delta_30s"]
KILL_REASON = {
    "microprice_dev_mom_60s": "rho=0.996 vs oir_mom_60s (keep oir_mom_60s)",
    "vol_adj_mom_60s": "rho=0.974 vs signed_rv_60s; also 0.849 vs panel ofi_60s (keep signed_rv_60s)",
    "top_book_delta_30s": "rho=0.878 vs wdi_mom_30s (keep wdi_mom_30s: lower panel corr 0.558 vs 0.622, higher OOS IC)",
}


def load_all():
    out = {}
    for f in sorted(REP.glob("screen_*_20250701_20250930_*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        out[j["prototype"]["name"]] = j
    return out


def round_head(hs):
    """Round head-stats floats for compact archival."""
    r = {}
    for k, v in hs.items():
        if isinstance(v, float):
            r[k] = round(v, 4)
        else:
            r[k] = v
    return r


def main():
    reps = load_all()
    print(f"reports loaded: {len(reps)}")

    admitted = []
    for name in ADMITTED:
        j = reps[name]
        dup = j["duplicate_check"]
        entry = {
            "name": name,
            "mechanism": j["prototype"].get("mechanism", ""),
            "max_abs_corr_vs_panel": round(dup["max_abs_corr"], 4),
            "nearest_panel_factor": dup["library_factor"],
            "horizons": {},
        }
        for h in j["horizons"]:
            entry["horizons"][str(h["horizon_s"])] = {
                "passed": h["passed"],
                "is_mean_ic": round(h["is_mean_ic"], 4),
                "is_t": round(h["is_t_stat_nw"], 2),
                "oos_mean_ic": round(h["oos_mean_ic"], 4),
                "oos_t": round(h["oos_t_stat_nw"], 2),
                "retention": round(h["retention"], 3),
                "head_stats": [round_head(s) for s in h["head_stats"]]
                if h["passed"] else [],
            }
        admitted.append(entry)

    killed = []
    for name in DEDUP_KILLED:
        j = reps[name]
        dup = j["duplicate_check"]
        passed_h = [h["horizon_s"] for h in j["horizons"] if h["passed"]]
        killed.append({
            "name": name,
            "screen_passed_horizons": passed_h,
            "max_abs_corr_vs_panel": round(dup["max_abs_corr"], 4),
            "killed_reason": KILL_REASON[name],
        })

    block = {"admitted": admitted, "dedup_killed": killed}
    (HERE / "round1_admitted_detail.json").write_text(
        json.dumps(block, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote round1_admitted_detail.json "
          f"({len(admitted)} admitted, {len(killed)} killed)")

    # ---- FAIL table for narrative ----
    fail_names = sorted(n for n in reps
                        if n not in ADMITTED and n not in DEDUP_KILLED
                        and not any(h["passed"] for h in reps[n]["horizons"]))
    print(f"\nFAIL protos: {len(fail_names)}")
    print(f"{'name':34s} {'best-h':>6s} {'IS_ic':>8s} {'IS_t':>7s} "
          f"{'OOS_ic':>8s} {'OOS_t':>7s} {'ret':>6s}  cause")
    for name in fail_names:
        j = reps[name]
        hs = j["horizons"]
        # best horizon by IS |t| (None stats => 0)
        b = max(hs, key=lambda h: abs(h["is_t_stat_nw"] or 0.0))
        is_t = b["is_t_stat_nw"] or 0.0
        oos_t = b["oos_t_stat_nw"] or 0.0
        is_ic = b["is_mean_ic"] or 0.0
        oos_ic = b["oos_mean_ic"] or 0.0
        ret = b["retention"] if b["retention"] is not None else float("nan")
        if abs(is_t) < 2.0:
            cause = "IS-dead"
        elif is_ic * oos_ic < 0:
            cause = f"sign-flip"
        elif ret < 0.5:
            cause = f"retention-collapse"
        else:
            cause = "oos-t-weak"
        # also check if ANY horizon had decent OOS
        best_oos = max(hs, key=lambda h: abs(h["oos_t_stat_nw"] or 0.0))
        oos_t_best = best_oos["oos_t_stat_nw"] or 0.0
        note = ""
        if abs(oos_t_best) >= 2.0 and not best_oos["passed"]:
            note = f" [near @ {best_oos['horizon_s']}s t={oos_t_best:.1f}]"
        dup = j["duplicate_check"]
        if dup.get("duplicated"):
            cause = f"DUP vs {dup['library_factor']} {dup['max_abs_corr']:.2f}"
        print(f"{name:34s} {b['horizon_s']:>6d} {is_ic:>+8.4f} "
              f"{is_t:>+7.2f} {oos_ic:>+8.4f} {oos_t:>+7.2f} "
              f"{ret:>6.3f}  {cause}{note}")


if __name__ == "__main__":
    main()
