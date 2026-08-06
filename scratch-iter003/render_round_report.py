"""Render a Chinese iter-003 round report from a candidates.json block.

Usage: python render_round_report.py [round_key]
  round_key defaults to the latest iter003_roundN in candidates.json.
Reads library/candidates.json + the round's manifests (for per-family table),
writes scratch-iter003/<round_key>_report.md.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # hft-autofactor/
LIB = ROOT / "library" / "candidates.json"


def latest_round_key(lib: dict) -> str:
    keys = [k for k in lib if re.fullmatch(r"iter003_round\d+", k)]
    return sorted(keys, key=lambda k: int(k.split("_round")[-1]))[-1]


def best_oos(horizons: dict) -> dict:
    passed = {h: d for h, d in horizons.items() if d["passed"]}
    if not passed:
        return {}
    h, d = max(passed.items(), key=lambda kv: abs(kv[1]["oos_t"]))
    return {"h": h, "oos_ic": d["oos_mean_ic"], "oos_t": d["oos_t"], "n_passed": len(passed)}


def family_map(round_key: str) -> dict:
    """name -> family letter, from manifest files (best-effort)."""
    fmap = {}
    n = int(round_key.split("_round")[-1])
    for mf in sorted((ROOT / "scratch-iter003").glob(f"manifest_iter003_R{n}*.json")):
        fam = mf.stem.split("_")[-1]  # e.g. R4A
        try:
            for e in json.loads(mf.read_text(encoding="utf-8")):
                fmap[e["name"]] = fam
        except Exception:
            pass
    return fmap


def render(round_key: str) -> str:
    lib = json.loads(LIB.read_text(encoding="utf-8"))
    b = lib[round_key]
    s = b["summary"]
    batch = b["batch"]
    n_round = round_key.split("_round")[-1]
    fmap = family_map(round_key)

    # per-family table (best-effort). batch.families keys look like
    # "R4A_spread_z_gates"; manifests key on the letter "R4A". Normalize to letter.
    fam_info = {}  # letter -> [label, count]
    for key, cnt in batch["families"].items():
        letter = key.split("_")[0]
        fam_info[letter] = [key, cnt]
    fam_admit = {L: 0 for L in fam_info}
    fam_killed = {L: set() for L in fam_info}
    for e in b["admitted"]:
        L = fmap.get(e["name"])
        if L in fam_admit:
            fam_admit[L] += 1
    for k in b["dedup"]["killed"]:
        L = fmap.get(k["name"])
        if L in fam_killed:
            fam_killed[L].add(k["name"])
    fam_pass = {L: fam_admit[L] + len(fam_killed[L]) for L in fam_info}

    out = []
    out.append(f"# iter-003 第{n_round}轮评估报告（588000，59 列宽表持续迭代）\n")
    out.append(f"日期：{b['generated_at']}　批次：{round_key}")
    out.append(f"前置：库 {s['library_total'] - s['admitted_new']}；本轮按经验定向挖 {len(fam_info)} 个方向。\n")
    out.append("## 一句话结果\n")
    out.append(f"{b['batch']['n_specs']} 个候选 → eval-v2 门槛过 {s['screen_pass']} → "
               f"对 {s['library_total'] - s['admitted_new']} 因子库 pooled Spearman 去重砍 {s['dedup_killed']} → "
               f"**入库 {s['admitted_new']}，库总计 {s['library_total']}**。\n")

    if fmap:
        out.append("## 批次构成\n")
        out.append("| 组 | 个数 | 过筛 | 入库 |")
        out.append("|---|---|---|---|")
        for L, (label, cnt) in fam_info.items():
            out.append(f"| {label} | {cnt} | {fam_pass.get(L, '?')} | {fam_admit.get(L, '?')} |")
        out.append("")

    out.append("## 入库因子（按最强 OOS |t| 排序）\n")
    rows = []
    for e in b["admitted"]:
        bo = best_oos(e["horizons"])
        rows.append((e, bo))
    rows.sort(key=lambda r: -abs(r[1].get("oos_t", 0)))
    out.append("| 因子 | 含义 | 过几个h | 最强 OOS IC（t） | 面板\\|ρ\\| |")
    out.append("|---|---|---|---|---|")
    for e, bo in rows:
        mech = e["mechanism"][:40] + ("…" if len(e["mechanism"]) > 40 else "")
        out.append(f"| {e['name']} | {mech} | {bo.get('n_passed','?')} | "
                   f"{bo['h']}s {bo['oos_ic']:+.4f}（{bo['oos_t']:+.2f}） | "
                   f"{e.get('max_abs_corr_vs_panel')} |")
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
    out.append(f"- library/candidates.json 键 `{round_key}`；"
               f"scratch-iter003/：reports_{n_round}（PASS JSON）、round{n_round}_admitted_detail.json、"
               f"round{n_round}_deathmap.json、{round_key}_corr.json/.txt、本报告。"
               f"服务器副本 /data/factor_lzt/iterations/，日志 /data/factor_lzt/logs/{round_key}.log。")

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    lib = json.loads(LIB.read_text(encoding="utf-8"))
    rk = sys.argv[1] if len(sys.argv) > 1 else latest_round_key(lib)
    md = render(rk)
    out = ROOT / "scratch-iter003" / f"{rk}_report.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out} ({len(md)} chars)")
