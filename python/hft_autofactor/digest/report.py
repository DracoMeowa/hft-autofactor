"""Digest builder: condense eval artifacts into JSON + Chinese markdown.

Pipeline (all inputs are read-only)::

    out_root/reports/eval_{first}_{last}.json   # IC tables + gates
    out_root/reports/trial_ledger.jsonl         # honest-N ledger
    out_root/parquet/dt={date}/factors.parquet  # panel (optional input)
        |
        v
    build_digest() -> dict
        |
        v
    report_dir/digest_{first}_{last}.json + .md

The digest is deliberately self-contained: it re-reads only artifacts the
eval stage already wrote, so it can run after every production run without
re-computing any IC.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..ingest import CANARY_FACTORS
from .correlations import DEFAULT_CORR_THRESHOLD, greedy_clusters, pairwise_spearman
from .coverage import coverage_report
from .data_quality import panel_quality, parquet_paths_for_dates, sample_factor_rows
from .ic_decay import decay_table
from .taxonomy import REASON_ZH, classify_outcomes

__all__ = [
    "build_digest",
    "find_eval_report",
    "ledger_counts",
    "render_markdown",
    "write_digest",
]


# --------------------------------------------------------------------- #
# artifact discovery                                                    #
# --------------------------------------------------------------------- #
def _report_dates(path: Path) -> list[str]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(d) for d in doc.get("dates") or []]


def find_eval_report(
    out_root: str | Path,
    *,
    dates: Sequence[str] | None = None,
    explicit: str | Path | None = None,
) -> Path:
    """Locate the eval JSON report to digest.

    Priority: ``explicit`` path > report whose covered dates best overlap
    ``dates`` > most recently modified.
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"eval report not found: {p}")
        return p

    candidates = sorted(Path(out_root).glob("reports/eval_*.json"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"no eval report (reports/eval_*.json) under {out_root}; "
            "run `hftaf eval` first"
        )
    if dates:
        wanted = set(str(d) for d in dates)

        def _score(p: Path) -> tuple[int, float]:
            return len(wanted & set(_report_dates(p))), p.stat().st_mtime

        return max(candidates, key=_score)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ledger_counts(path: str | Path) -> dict:
    """Count ledger trials overall and per stage (no side effects)."""
    path = Path(path)
    by_stage: dict[str, int] = {}
    total = 0
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    stage = str(json.loads(line).get("stage") or "unknown")
                except json.JSONDecodeError:
                    stage = "unparseable"
                by_stage[stage] = by_stage.get(stage, 0) + 1
    return {"total": total, "by_stage": by_stage}


# --------------------------------------------------------------------- #
# builder                                                               #
# --------------------------------------------------------------------- #
def build_digest(
    out_root: str | Path,
    *,
    dates: Sequence[str] | None = None,
    eval_report: str | Path | None = None,
    max_rows: int = 200_000,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    include_panel: bool = True,
) -> dict:
    """Assemble the full digest document (a JSON-serializable dict)."""
    out_root = Path(out_root)
    report_path = find_eval_report(out_root, dates=dates, explicit=eval_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    rep_dates = [str(d) for d in report.get("dates") or []]
    eff_dates = [str(d) for d in dates] if dates else rep_dates
    factors_evaluated = [str(f) for f in report.get("factors") or []]
    horizons = sorted(int(h) for h in report.get("horizons_s") or [])

    decay = decay_table(report)

    # ---- panel pass (data quality + correlations) --------------------- #
    dq: dict | None = None
    corr_factors: list[str] = []
    corr_matrix: dict[str, dict[str, float]] = {}
    clusters: list[dict] = []
    n_sample_rows = 0
    paths = (
        parquet_paths_for_dates(out_root, eff_dates)
        if include_panel and eff_dates
        else []
    )
    if paths:
        wanted = [f for f in factors_evaluated if f not in CANARY_FACTORS]
        dq = panel_quality(paths, factor_cols=wanted or None)
        # NaN-by-design columns cannot be ranked; exclude from correlations
        corr_factors = [
            f
            for f in wanted
            if dq["factor_nan_rates"].get(f, 1.0) < 0.999
        ]
        if len(corr_factors) >= 2:
            sample = sample_factor_rows(paths, corr_factors, max_rows=max_rows)
            usable = [f for f in corr_factors if f in sample.columns]
            if len(usable) >= 2 and sample.height > 1:
                corr = pairwise_spearman(sample, usable)
                corr_matrix = {}
                for (a, b), v in corr.items():
                    corr_matrix.setdefault(a, {})[b] = v
                    corr_matrix.setdefault(b, {})[a] = v
                clusters = greedy_clusters(corr, threshold=corr_threshold)
                n_sample_rows = sample.height
                corr_factors = usable

    taxonomy = classify_outcomes(
        report,
        nan_rates=(dq or {}).get("factor_nan_rates"),
        nan_rates_by_exchange=(dq or {}).get("factor_nan_rates_by_exchange"),
    )

    coverage = coverage_report(
        factors_evaluated, decay, report.get("stage1_screen") or []
    )

    digest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "out_root": str(out_root),
        "eval_report": str(report_path),
        "dates": eff_dates,
        "factors": factors_evaluated,
        "horizons_s": horizons,
        "n_trials": ledger_counts(out_root / "reports" / "trial_ledger.jsonl"),
        "panel_available": bool(paths),
        "ic_decay": decay,
        "taxonomy": taxonomy,
        "correlations": {
            "threshold": corr_threshold,
            "n_sample_rows": n_sample_rows,
            "factors": corr_factors,
            "matrix": corr_matrix,
            "clusters": clusters,
        },
        "coverage": coverage,
        "data_quality": dq
        if dq is not None
        else {
            "available": False,
            "reason": "no parquet partitions found for the report dates",
        },
    }
    return digest


# --------------------------------------------------------------------- #
# serialization                                                         #
# --------------------------------------------------------------------- #
def _sanitize(obj: Any) -> Any:
    """NaN/inf -> null, floats rounded to 6 dp (JSON-friendly)."""
    if isinstance(obj, Mapping):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return round(obj, 6)
    return obj


def write_digest(digest: Mapping, report_dir: str | Path) -> tuple[Path, Path]:
    """Write ``digest_{first}_{last}.json`` + ``.md``; returns both paths."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    dates = digest.get("dates") or []
    stamp = f"{dates[0]}_{dates[-1]}" if dates else "noeval"
    json_path = report_dir / f"digest_{stamp}.json"
    md_path = report_dir / f"digest_{stamp}.md"
    json_path.write_text(
        json.dumps(_sanitize(digest), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(digest), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------- #
# markdown rendering (Chinese insight report)                           #
# --------------------------------------------------------------------- #
def _fmt_ic_cell(cell: Mapping | None) -> str:
    if not cell or cell.get("mean_ic") is None:
        return "—"
    try:
        ic = float(cell["mean_ic"])
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(ic):
        return "—"
    t = cell.get("t_stat_nw")
    t_txt = ""
    if t is not None:
        try:
            t = float(t)
            if math.isfinite(t):
                t_txt = f" (t={t:.2f})"
        except (TypeError, ValueError):
            pass
    return f"{ic:+.4f}{t_txt}"


def _fmt_pct(x: Any) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_num(x: Any, nd: int = 4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def _decay_section(digest: Mapping, horizons: Sequence[int]) -> list[str]:
    lines = ["## 1. IC 衰减与半衰期", ""]
    header = "| 因子 | " + " | ".join(f"{h}s" for h in horizons) + \
             " | 峰值 horizon | 半衰期 |"
    sep = "|---" * (len(horizons) + 3) + "|"
    lines += [header, sep]
    for row in digest.get("ic_decay", []):
        cells = [
            _fmt_ic_cell((row.get("horizons") or {}).get(h)) for h in horizons
        ]
        peak = row.get("peak_horizon_s")
        half = row.get("half_life_s")
        peak_txt = f"{peak}s" if peak is not None else "—"
        half_txt = f"{half}s" if half is not None else ">最长horizon"
        lines.append(
            f"| {row['factor']} | " + " | ".join(cells) +
            f" | {peak_txt} | {half_txt} |"
        )
    lines += [
        "",
        "> 单元格为 mean RankIC（括号内为 Newey-West t）；“—”表示无有效观测。",
        "> 半衰期 = 峰值之后 |IC| 首次跌破峰值一半的 horizon；"
        "“>最长horizon”表示衰减比最长 horizon 更慢（长记忆信号）。",
        "",
    ]
    return lines


def _taxonomy_section(digest: Mapping) -> list[str]:
    rows = digest.get("taxonomy", [])
    n = len(rows)
    n_s1 = sum(1 for r in rows if r.get("stage1_passed"))
    n_wf = sum(1 for r in rows if r.get("wf_passed"))
    n_both = sum(1 for r in rows if r.get("combined_passed"))
    lines = [
        "## 2. 通过 / 失败分类",
        "",
        f"共 {n} 个（因子 × horizon）组合：Stage-1 通过 **{n_s1}**，"
        f"walk-forward 通过（≥50% 折）**{n_wf}**，两关全过 **{n_both}**。",
        "",
    ]

    passed = [r for r in rows if r.get("combined_passed")]
    if passed:
        lines += [
            "### 全过组合",
            "",
            "| 因子 | horizon | Stage-1 | WF 通过率 |",
            "|---|---|---|---|",
        ]
        for r in passed:
            rate = r.get("wf_pass_rate")
            lines.append(
                f"| {r['factor']} | {r['horizon_s']}s | ✅ | {_fmt_pct(rate)} |"
            )
        lines.append("")

    # failure reason summary (only failed combos)
    reason_combos: dict[str, list[str]] = {}
    nan_factors: set[str] = set()
    for r in rows:
        if r.get("combined_passed"):
            continue
        if r.get("nan_by_design"):
            nan_factors.add(r["factor"])
        for key in r.get("failure_reasons", []):
            reason_combos.setdefault(key, []).append(
                f"{r['factor']}@{r['horizon_s']}s"
            )
    if reason_combos:
        lines += [
            "### 失败原因统计",
            "",
            "| 原因 | 组合数 | 示例（最多 8 个） |",
            "|---|---|---|",
        ]
        for key in sorted(reason_combos, key=lambda k: -len(reason_combos[k])):
            combos = reason_combos[key]
            sample = ", ".join(combos[:8]) + ("…" if len(combos) > 8 else "")
            lines.append(f"| {REASON_ZH.get(key, key)} | {len(combos)} | {sample} |")
        lines.append("")
    if nan_factors:
        lines += [
            "NaN-by-design 因子（无法评估，属数据通道问题而非因子无效）："
            + "、".join(sorted(nan_factors)),
            "",
        ]
    return lines


def _correlation_section(digest: Mapping) -> list[str]:
    corr = digest.get("correlations", {}) or {}
    threshold = corr.get("threshold", DEFAULT_CORR_THRESHOLD)
    n_sample = corr.get("n_sample_rows", 0)
    clusters = corr.get("clusters", [])
    lines = ["## 3. 因子相关性簇（冗余族）", ""]
    if not corr.get("factors"):
        lines += ["（面板不可用或可用因子不足 2 个，未计算相关性。）", ""]
        return lines
    lines.append(
        f"抽样 {n_sample} 行（跨日等距抽样），Spearman 相关；"
        f"|ρ| ≥ {threshold:.2f} 贪心单链接聚簇。"
    )
    lines.append("")
    if clusters:
        for c in clusters:
            members = ", ".join(c["members"])
            lines.append(
                f"- **{c['name']}**：{members}"
                f"（族内平均 |ρ| = {_fmt_num(c.get('mean_abs_corr'), 3)}）"
            )
        lines += [
            "",
            "> 同族因子信息高度重叠：下一轮假设应优先跨族变异，"
            "或在族内做正交化/择优选一。",
            "",
        ]
    else:
        lines += [
            f"在阈值 {threshold:.2f} 下未发现冗余簇——12 个库因子信息上相互独立。",
            "",
        ]
    return lines


def _coverage_section(digest: Mapping) -> list[str]:
    cov = digest.get("coverage", {}) or {}
    lines = ["## 4. 覆盖缺口与机会提示", "", "### 信息维度覆盖", ""]
    lines += [
        "| 维度 | 已覆盖因子 | 状态 |",
        "|---|---|---|",
    ]
    for dim, info in (cov.get("dimension_coverage") or {}).items():
        covered_by = info.get("covered_by") or []
        if not info.get("covered"):
            status = "❌ 缺口"
        elif info.get("thin"):
            status = "⚠️ 薄弱（仅 1 因子）"
        else:
            status = "✅ 已覆盖"
        lines.append(
            f"| {info.get('zh', dim)} | {', '.join(covered_by) or '—'} | {status} |"
        )
    lines.append("")

    unmapped = cov.get("unmapped_factors") or []
    if unmapped:
        lines.append(f"未映射到已知维度的因子（新挖掘）：{', '.join(unmapped)}")
        lines.append("")

    weakest = cov.get("weakest_horizons") or []
    if weakest:
        lines += ["### 最弱 horizon（按跨因子平均 |IC| 升序）", ""]
        lines += [
            "| horizon | 平均 abs(IC) | 有 IC 的因子数 | Stage-1 通过/总数 |",
            "|---|---|---|---|",
        ]
        for w in weakest:
            lines.append(
                f"| {w['horizon_s']}s | {_fmt_num(w.get('mean_abs_ic'))} "
                f"| {w.get('n_factors_with_ic', 0)} "
                f"| {w.get('stage1_pass', 0)}/{w.get('stage1_combos', 0)} |"
            )
        lines += [
            "",
            "> 排名靠前的 horizon 是当前证据最弱的预测窗口，"
            "可作为下一轮因子提案的靶点。",
            "",
        ]

    hints = cov.get("opportunity_hints") or []
    if hints:
        lines += ["### 机会提示（供提案参考）", ""]
        dim_zh = {
            dim: info.get("zh", dim)
            for dim, info in (cov.get("dimension_coverage") or {}).items()
        }
        for h in hints:
            lines.append(
                f"- **{dim_zh.get(h['dimension'], h['dimension'])}**："
                f"{h['hint_zh']}"
            )
        lines.append("")
    return lines


def _dq_section(digest: Mapping) -> list[str]:
    dq = digest.get("data_quality") or {}
    lines = ["## 5. 数据质量", ""]
    if not dq or dq.get("available") is False:
        reason = (dq or {}).get("reason", "面板不可用")
        lines += [f"（{reason}，本节跳过。）", ""]
        return lines

    n_rows = dq.get("n_rows", 0)
    lines.append(
        f"面板行数：**{n_rows:,}**（{dq.get('n_partitions', 0)} 个日分区）；"
        "分交易所行数："
        + "，".join(
            f"{exch} {cnt:,}"
            for exch, cnt in sorted((dq.get("n_rows_by_exchange") or {}).items())
        )
    )
    lines += ["", "### flag 位频率", "", "| bit | 名称 | 频率 |", "|---|---|---|"]
    bit_names = {
        0: "book_unsynced",
        1: "seq_gap_before",
        2: "iopv_invalid",
        3: "one_sided_book",
    }
    for bit, name in bit_names.items():
        rate = (dq.get("flag_bit_rates") or {}).get(name)
        lines.append(f"| bit{bit} | {name} | {_fmt_pct(rate)} |")
    lines.append(f"| — | flags==0（干净行） | {_fmt_pct(dq.get('clean_rows_rate'))} |")
    lines += [
        "",
        f"一边倒书（flag bit3）：{_fmt_pct(dq.get('one_sided_book_rate'))}；"
        f"价格侧缺失交叉核对（bid1/ask1 空或非正）："
        f"{_fmt_pct(dq.get('quote_side_missing_rate'))}",
        "",
    ]

    absent = dq.get("absent_label_rates") or {}
    if absent:
        lines += [
            "### ABSENT 标签率（按 horizon）",
            "",
            "| 标签 | ABSENT 率 |",
            "|---|---|",
        ]
        for lbl in sorted(absent):
            lines.append(f"| {lbl} | {_fmt_pct(absent[lbl])} |")
        lines += [
            "",
            "> 收盘前 horizon 越长 ABSENT 越多属按设计行为（标签跨不过收盘）。",
            "",
        ]

    nan_rates = dq.get("factor_nan_rates") or {}
    by_exch = dq.get("factor_nan_rates_by_exchange") or {}
    if nan_rates:
        exchanges = sorted(by_exch)
        header = "| 因子 | 总体 NaN | " + " | ".join(
            f"{e} NaN" for e in exchanges
        ) + " |"
        lines += ["### 因子 NaN 率", "", header, "|---" * (2 + len(exchanges)) + "|"]
        for f in sorted(nan_rates):
            row = [f, _fmt_pct(nan_rates[f])]
            for e in exchanges:
                row.append(_fmt_pct(by_exch[e].get(f)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return lines


def render_markdown(digest: Mapping) -> str:
    """Render the Chinese insight report for one digest document."""
    horizons = [
        int(h) for h in (digest.get("horizons_s") or [15, 30, 60, 300, 900])
    ]
    nt = digest.get("n_trials", {}) or {}
    by_stage = nt.get("by_stage", {}) or {}
    lines = [
        "# hft-autofactor 评估摘要（digest）",
        "",
        f"- 生成时间：{digest.get('generated_at', '—')}",
        f"- 数据根目录：`{digest.get('out_root', '—')}`",
        f"- 评估报告：`{digest.get('eval_report', '—')}`",
        f"- 日期范围：{digest.get('dates', ['—'])[0]} … "
        f"{digest.get('dates', ['—'])[-1]}"
        f"（{len(digest.get('dates', []))} 天）",
        f"- 试验登记（honest N）：总计 {nt.get('total', 0)}"
        + "".join(f"，{k} {v}" for k, v in sorted(by_stage.items())),
        f"- 面板可用：{'是' if digest.get('panel_available') else '否'}",
        "",
    ]
    lines += _decay_section(digest, horizons)
    lines += _taxonomy_section(digest)
    lines += _correlation_section(digest)
    lines += _coverage_section(digest)
    lines += _dq_section(digest)
    lines += [
        "---",
        "*本报告由 hftaf-digest 自动生成，供下一轮因子假设演化参考。*",
        "",
    ]
    return "\n".join(lines)
