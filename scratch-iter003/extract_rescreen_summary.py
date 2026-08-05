"""Extract a compact summary from rescreen_reports.json for the archive.

Prints, per prototype: per-horizon IS/OOS IC + t + retention + pass flag, and
for PASS horizons the head long/short stats (tau 1/5/10, gross+net).
"""
import json

with open(r"D:\claude\Quant_works\hft-autofactor\scratch-iter003\rescreen_reports.json", encoding="utf-8") as f:
    reports = json.load(f)

probe = next(iter(reports.values()))
hz_probe = probe.get("horizons")
if isinstance(hz_probe, list) and hz_probe:
    print("horizon entry keys:", sorted(hz_probe[0].keys()))
    print("split:", json.dumps(probe.get("split"), ensure_ascii=False))
print("=" * 70)

for name in sorted(reports):
    rep = reports[name]
    hz = rep.get("horizons") or []
    if isinstance(hz, dict):
        hz = [{"horizon": k, **v} for k, v in hz.items()]
    hz = sorted(hz, key=lambda e: int(e.get("horizon", e.get("h", 0))))
    passed = rep.get("passed")
    print(f"\n### {name}  status={rep.get('status')} passed={passed}")
    for e in hz:
        h = e.get("horizon", e.get("h"))
        print(
            f"  h={h} | "
            + " | ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in e.items() if k != "horizon")
        )
