"""Summarize iter-003 round-1 screen reports (local copies).

Prints per-proto per-horizon IC/t/retention for all PASS protos and a
grouped FAIL list, to pick cluster representatives and feed the archive.
"""
import json
from pathlib import Path

REP = Path(r"D:\claude\Quant_works\hft-autofactor\scratch-iter003\reports_round1")


def main() -> None:
    rows = []
    for f in sorted(REP.glob("screen_*_20250701_20250930_*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        name = j["prototype"]["name"]
        ok = any(h.get("passed") for h in (j.get("horizons") or []))
        rows.append((name, ok, j))
    print(f"reports: {len(rows)}")

    passed = [(n, j) for n, ok, j in rows if ok]
    failed = [(n, j) for n, ok, j in rows if not ok]
    print(f"PASS {len(passed)}  FAIL {len(failed)}\n")

    print("=== PASS protos (per-horizon detail) ===")
    for name, j in sorted(passed):
        dup = j.get("duplicate_check") or {}
        mc = dup.get("max_abs_corr")
        mc_s = f"{mc:.3f}" if isinstance(mc, (int, float)) else str(mc)
        hs = j.get("horizons") or []
        ph = [str(h["horizon_s"]) for h in hs if h.get("passed")]
        print(f"\n{name}  max|corr|={mc_s} lib={dup.get('library_factor')}"
              f"  passed={'+'.join(ph)}")
        for h in hs:
            mark = "PASS" if h.get("passed") else "fail"
            print(f"  {h['horizon_s']:4d}s {mark} IS ic={h['is_mean_ic']:+.4f}"
                  f" t={h['is_t_stat_nw']:+6.2f} | OOS ic={h['oos_mean_ic']:+.4f}"
                  f" t={h['oos_t_stat_nw']:+6.2f} ret={h['retention']:.3f}")

    print("\n=== FAIL protos (first reasons) ===")
    for name, j in sorted(failed):
        reasons = j.get("reasons") or []
        dup = j.get("duplicate_check") or {}
        mc = dup.get("max_abs_corr")
        if dup.get("duplicated"):
            reasons = [f"dup {mc:.3f} vs {dup.get('library_factor')}"] + reasons
        print(f"{name:34s} {'; '.join(reasons)[:150]}")


if __name__ == "__main__":
    main()
