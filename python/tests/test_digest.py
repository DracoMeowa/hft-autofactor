"""Tests for the digest subpackage on synthetic eval artifacts.

No C++ engine, no real data: a hand-built eval JSON report (same schema as
``run_eval_stage``), a JSONL trial ledger, and small engineered parquet day
partitions under a tmp out_root exercise every digest section.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_autofactor.digest import (
    build_digest,
    classify_outcomes,
    coverage_report,
    decay_table,
    find_eval_report,
    greedy_clusters,
    ledger_counts,
    pairwise_spearman,
    panel_quality,
    parquet_paths_for_dates,
    render_markdown,
    sample_factor_rows,
    write_digest,
)
from hft_autofactor.digest.cli import main as digest_main

HORIZONS = (15, 30, 60, 300, 900)
FACTORS = (
    "oir",
    "wdi",
    "book_slope",
    "ofi_60s",
    "order_arrival_60s",
    "cancel_ratio_60s",
)


# --------------------------------------------------------------------- #
# synthetic eval artifacts                                              #
# --------------------------------------------------------------------- #
def _stats_row(factor, h, mean_ic, icir, t, n_obs=160, win_rate=0.6):
    return {
        "factor": factor,
        "horizon_s": h,
        "n_obs": n_obs,
        "mean_ic": mean_ic,
        "ic_std": 0.04 if mean_ic is not None else None,
        "icir": icir,
        "t_stat_nw": t,
        "n_eff": 120.0 if n_obs else 0.0,
        "win_rate": win_rate if mean_ic is not None else None,
    }


def _s1_row(factor, h, mean_ic, icir, t, *, fdr_pass, passed,
            noise_floor=0.005, n_obs=160):
    return {
        "factor": factor,
        "horizon_s": h,
        "n_obs": n_obs,
        "mean_ic": mean_ic,
        "icir": icir,
        "t_stat_nw": t,
        "n_trials": 60,
        "t_hurdle_min": 3.0,
        "noise_floor": noise_floor,
        "p_value": 0.001 if passed else 0.2,
        "fdr_pass": fdr_pass,
        "passed": passed,
    }


def make_eval_report() -> dict:
    """A controlled eval report: one full pass, gate-specific failures,
    NaN-by-design factors, and a clear IC decay shape for oir."""
    stats = []
    s1 = []
    # --- oir: peak at 60s, half at 300s ------------------------------- #
    oir_curve = {15: (0.03, 1.5, 2.8), 30: (0.04, 1.0, 2.1),
                 60: (0.05, 1.25, 4.5), 300: (0.02, 0.45, 1.9),
                 900: (0.008, 0.2, 0.8)}
    for h, (ic, icir, t) in oir_curve.items():
        stats.append(_stats_row("oir", h, ic, icir, t))
    s1.append(_s1_row("oir", 15, 0.03, 1.5, 2.8, fdr_pass=True, passed=False))
    s1.append(_s1_row("oir", 30, 0.04, 1.0, 2.1, fdr_pass=True, passed=False))
    s1.append(_s1_row("oir", 60, 0.05, 1.25, 4.5, fdr_pass=True, passed=True))
    s1.append(_s1_row("oir", 300, 0.02, 0.45, 1.9, fdr_pass=False, passed=False,
                      noise_floor=0.006))
    s1.append(_s1_row("oir", 900, 0.008, 0.2, 0.8, fdr_pass=False, passed=False,
                      noise_floor=0.004))

    # --- wdi: redundant with oir, fails FDR at 60s -------------------- #
    for h, (ic, icir, t) in oir_curve.items():
        stats.append(_stats_row("wdi", h, ic * 0.96, icir, t))
    s1.append(_s1_row("wdi", 60, 0.048, 1.2, 4.2, fdr_pass=False, passed=False))

    # --- book_slope: inverted oir family ------------------------------ #
    for h, (ic, icir, t) in oir_curve.items():
        stats.append(_stats_row("book_slope", h, -ic * 0.9, icir, t))
    s1.append(_s1_row("book_slope", 60, -0.045, 1.1, 3.9, fdr_pass=True,
                      passed=True))

    # --- ofi_60s: weak everywhere -------------------------------------- #
    ofi_curve = {15: 0.012, 30: 0.011, 60: 0.010, 300: 0.009, 900: 0.008}
    for h, ic in ofi_curve.items():
        stats.append(_stats_row("ofi_60s", h, ic, 0.3, 1.1))
        s1.append(_s1_row("ofi_60s", h, ic, 0.3, 1.1, fdr_pass=False,
                          passed=False))

    # --- order_arrival_60s / cancel_ratio_60s: NaN-by-design on SSE ---- #
    for factor in ("order_arrival_60s", "cancel_ratio_60s"):
        for h in HORIZONS:
            stats.append(_stats_row(factor, h, None, None, None, n_obs=0,
                                    win_rate=None))
            s1.append(_s1_row(factor, h, None, None, None, fdr_pass=False,
                              passed=False, noise_floor=None, n_obs=0))

    walk_forward = [
        # oir@60: 1/2 folds pass -> wf pass at the 50% majority rule
        {"factor": "oir", "horizon_s": 60, "fold": 0,
         "train_dates": ["20250701"], "test_dates": ["20250702"],
         "is_mean_ic": 0.05, "oos_mean_ic": 0.03, "oos_t_stat_nw": 2.5,
         "passed": True,
         "details": {"horizon_s": 60, "is_mean_ic": 0.05, "oos_mean_ic": 0.03,
                     "retention": 0.6, "retention_ok": True, "sign_ok": True,
                     "oos_t_ok": True, "win_rate_ok": True, "level_ok": True}},
        {"factor": "oir", "horizon_s": 60, "fold": 1,
         "train_dates": ["20250701"], "test_dates": ["20250702"],
         "is_mean_ic": 0.05, "oos_mean_ic": 0.01, "oos_t_stat_nw": 0.9,
         "passed": False,
         "details": {"horizon_s": 60, "is_mean_ic": 0.05, "oos_mean_ic": 0.01,
                     "retention": 0.2, "retention_ok": False, "sign_ok": True,
                     "oos_t_ok": False, "win_rate_ok": True, "level_ok": False}},
        # wdi@60: fails stage1 anyway; both folds decay
        {"factor": "wdi", "horizon_s": 60, "fold": 0,
         "train_dates": ["20250701"], "test_dates": ["20250702"],
         "is_mean_ic": 0.048, "oos_mean_ic": 0.01, "oos_t_stat_nw": 1.0,
         "passed": False,
         "details": {"retention": 0.2, "retention_ok": False, "sign_ok": True,
                     "oos_t_ok": False, "win_rate_ok": True, "level_ok": False}},
        # book_slope@60: stage1 passed, wf folds both pass -> combined pass
        {"factor": "book_slope", "horizon_s": 60, "fold": 0,
         "train_dates": ["20250701"], "test_dates": ["20250702"],
         "is_mean_ic": -0.045, "oos_mean_ic": -0.03, "oos_t_stat_nw": 2.6,
         "passed": True,
         "details": {"retention": 0.66, "retention_ok": True, "sign_ok": True,
                     "oos_t_ok": True, "win_rate_ok": True, "level_ok": True}},
    ]

    return {
        "generated_at": "2026-08-04T00:00:00",
        "dates": ["20250701", "20250702"],
        "factors": list(FACTORS),
        "horizons_s": list(HORIZONS),
        "stats": stats,
        "cross_section_stats": [],
        "noise_floors": [],
        "stage1_screen": s1,
        "walk_forward": walk_forward,
        "n_trials_total": 62,
    }


def make_partition(date: str, exchange: str, *, n: int = 400, seed: int,
                   order_arrival_nan: bool = False) -> pl.DataFrame:
    """One engineered day partition.

    oir ~ N(0,1); wdi = oir + tiny noise; book_slope = -0.95*oir + noise
    (one redundant depth family); ofi independent; order_arrival optionally
    all-NaN (the SSE NaN-by-design case); cancel_ratio always all-NaN.
    Flags: i % 10 == 0 -> ONE_SIDED_BOOK (with zeroed quotes);
    i % 7 == 0 -> IOPV_INVALID.
    """
    rng = np.random.default_rng(seed)
    oir = rng.standard_normal(n)
    wdi = oir + 0.05 * rng.standard_normal(n)
    book_slope = -0.95 * oir + 0.05 * rng.standard_normal(n)
    ofi = rng.standard_normal(n)
    order_arrival = (
        np.full(n, np.nan) if order_arrival_nan else rng.standard_normal(n)
    )
    cancel = np.full(n, np.nan)

    flags = np.zeros(n, dtype=np.uint32)
    one_sided = np.arange(n) % 10 == 0
    iopv_bad = np.arange(n) % 7 == 0
    flags[one_sided] |= 8
    flags[iopv_bad] |= 4

    labels_15 = rng.standard_normal(n)
    labels_15[np.arange(n) % 10 == 0] = np.nan          # 10% ABSENT
    labels_900 = rng.standard_normal(n)
    labels_900[np.arange(n) % 5 < 2] = np.nan            # 40% ABSENT

    def col_or_none(arr):
        return pl.Series(arr, dtype=pl.Float64)

    return pl.DataFrame(
        {
            "date": [date] * n,
            "exchange": [exchange] * n,
            "instrument": [f"ETF{i % 4:02d}" for i in range(n)],
            "ts_ms": [34_200_000 + 3000 * i for i in range(n)],
            "snap_seq": list(range(1000, 1000 + n)),
            "flags": flags,
            "mid_px": [4.0] * n,
            "last_px": [4.0] * n,
            "bid1_px": np.where(one_sided, 0.0, 3.999),
            "ask1_px": np.where(one_sided, 0.0, 4.001),
            "bid1_qty": [10000] * n,
            "ask1_qty": [8000] * n,
            "depth_bid5": [50000] * n,
            "depth_ask5": [42000] * n,
            "oir": col_or_none(oir),
            "wdi": col_or_none(wdi),
            "book_slope": col_or_none(book_slope),
            "ofi_60s": col_or_none(ofi),
            "order_arrival_60s": col_or_none(order_arrival),
            "cancel_ratio_60s": col_or_none(cancel),
            "fwd_mid_ret_15s": col_or_none(labels_15),
            "fwd_mid_ret_900s": col_or_none(labels_900),
            "fwd_last_ret_15s": col_or_none(labels_15.copy()),
            "channel": [1] * n,
        }
    )


@pytest.fixture
def digest_root(tmp_path) -> Path:
    """tmp out_root with eval report, ledger, and two day partitions."""
    out_root = tmp_path / "factor_lzt"
    reports = out_root / "reports"
    reports.mkdir(parents=True)

    report = make_eval_report()
    (reports / "eval_20250701_20250702.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(reports / "trial_ledger.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"factor": "oir", "horizon_s": 60, "params": {},
                             "stage": "stage1", "metrics": {}}) + "\n")
        fh.write(json.dumps({"factor": "wdi", "horizon_s": 60, "params": {},
                             "stage": "stage1", "metrics": {}}) + "\n")
        fh.write(json.dumps({"factor": "oir", "horizon_s": 60,
                             "params": {"fold": 0},
                             "stage": "stage2_walkforward",
                             "metrics": {}}) + "\n")

    for date, exch, oa_nan in (("20250701", "sse", True),
                               ("20250702", "szse", False)):
        part = make_partition(date, exch, seed=hash(date) % 2**32,
                              order_arrival_nan=oa_nan)
        path = out_root / "parquet" / f"dt={date}" / "factors.parquet"
        path.parent.mkdir(parents=True)
        part.write_parquet(path)
    return out_root


def _combo(taxonomy, factor, h):
    for row in taxonomy:
        if row["factor"] == factor and row["horizon_s"] == h:
            return row
    raise AssertionError(f"{factor}@{h} not in taxonomy")


# --------------------------------------------------------------------- #
# 1. IC decay + half-life                                               #
# --------------------------------------------------------------------- #
def test_decay_peak_and_half_life():
    decay = decay_table(make_eval_report())
    by_factor = {r["factor"]: r for r in decay}

    oir = by_factor["oir"]
    assert oir["peak_horizon_s"] == 60
    assert oir["peak_abs_ic"] == pytest.approx(0.05)
    # decay side of the peak: 300s (0.02) is the first below 0.025
    assert oir["half_life_s"] == 300
    assert oir["horizons"][60]["mean_ic"] == pytest.approx(0.05)
    assert oir["horizons"][15]["t_stat_nw"] == pytest.approx(2.8)

    # negative factor: |IC| peak semantics
    bs = by_factor["book_slope"]
    assert bs["peak_horizon_s"] == 60
    assert bs["peak_abs_ic"] == pytest.approx(0.045)

    # NaN-by-design factors: no finite IC anywhere
    cr = by_factor["cancel_ratio_60s"]
    assert cr["peak_horizon_s"] is None
    assert cr["half_life_s"] is None


def test_decay_handles_json_nulls():
    report = {
        "horizons_s": [15, 30],
        "stats": [
            {"factor": "x", "horizon_s": 15, "n_obs": 10, "mean_ic": None,
             "icir": None, "t_stat_nw": None, "win_rate": None},
            {"factor": "x", "horizon_s": 30, "n_obs": 10, "mean_ic": 0.02,
             "icir": 0.8, "t_stat_nw": 3.1, "win_rate": 0.6},
        ],
    }
    decay = decay_table(report)
    assert len(decay) == 1
    assert decay[0]["peak_horizon_s"] == 30
    assert decay[0]["half_life_s"] is None  # peak at last horizon


def test_decay_half_life_none_when_still_rising():
    report = {
        "horizons_s": [15, 30, 60],
        "stats": [
            {"factor": "slow", "horizon_s": h, "n_obs": 10, "mean_ic": ic,
             "t_stat_nw": 3.0, "icir": 1.0, "win_rate": 0.6}
            for h, ic in ((15, 0.01), (30, 0.02), (60, 0.04))
        ],
    }
    decay = decay_table(report)
    assert decay[0]["peak_horizon_s"] == 60
    assert decay[0]["half_life_s"] is None


# --------------------------------------------------------------------- #
# 2. pass/fail taxonomy                                                 #
# --------------------------------------------------------------------- #
def test_taxonomy_without_panel():
    tax = classify_outcomes(make_eval_report())

    # the full pass
    oir60 = _combo(tax, "oir", 60)
    assert oir60["stage1_passed"] and oir60["wf_passed"]
    assert oir60["combined_passed"] is True
    assert oir60["failure_reasons"] == []

    # book_slope@60: stage1 pass + wf pass (1/1 folds)
    assert _combo(tax, "book_slope", 60)["combined_passed"] is True

    # oir@30 fails ONLY on the t hurdle; also thin-margin cost suspect?
    oir30 = _combo(tax, "oir", 30)
    assert oir30["combined_passed"] is False
    assert "t_below_hurdle" in oir30["failure_reasons"]
    assert "below_ic_level" not in oir30["failure_reasons"]

    # oir@15: t hurdle + short-horizon cost suspect (0.03 < 2*0.02)
    oir15 = _combo(tax, "oir", 15)
    assert "t_below_hurdle" in oir15["failure_reasons"]
    assert "cost_dominated_suspect" in oir15["failure_reasons"]

    # oir@900: level + icir + t + fdr + horizon mismatch (peak at 60)
    oir900 = _combo(tax, "oir", 900)
    for key in ("below_ic_level", "low_icir", "t_below_hurdle",
                "fdr_not_passed", "horizon_mismatch"):
        assert key in oir900["failure_reasons"], key

    # NaN-by-design via n_obs == 0 (no panel given)
    cr = _combo(tax, "cancel_ratio_60s", 60)
    assert cr["nan_by_design"] is True
    assert cr["failure_reasons"] == ["nan_by_design"]

    # wdi@60: fails FDR only (t/icir/level all clear)
    wdi60 = _combo(tax, "wdi", 60)
    assert wdi60["failure_reasons"] == ["fdr_not_passed"]


def test_taxonomy_with_panel_nan_rates(digest_root):
    dq = panel_quality(
        parquet_paths_for_dates(digest_root, ["20250701", "20250702"]),
        factor_cols=list(FACTORS),
    )
    tax = classify_outcomes(
        make_eval_report(),
        nan_rates=dq["factor_nan_rates"],
        nan_rates_by_exchange=dq["factor_nan_rates_by_exchange"],
    )
    # cancel_ratio: 100% NaN everywhere
    assert _combo(tax, "cancel_ratio_60s", 60)["failure_reasons"] == [
        "nan_by_design"
    ]
    # order_arrival: NaN only on SSE -> per-exchange detection
    assert _combo(tax, "order_arrival_60s", 60)["nan_by_design"] is True
    # oir unaffected
    assert _combo(tax, "oir", 60)["combined_passed"] is True


# --------------------------------------------------------------------- #
# 3. correlation clusters                                               #
# --------------------------------------------------------------------- #
def test_pairwise_spearman_and_clusters(digest_root):
    paths = parquet_paths_for_dates(digest_root, ["20250701", "20250702"])
    sample = sample_factor_rows(
        paths, ["oir", "wdi", "book_slope", "ofi_60s"], max_rows=10_000
    )
    assert sample.height > 100
    corr = pairwise_spearman(sample, ["oir", "wdi", "book_slope", "ofi_60s"])
    assert corr[("oir", "wdi")] > 0.9
    assert corr[("oir", "book_slope")] < -0.9
    assert abs(corr[("oir", "ofi_60s")]) < 0.3

    clusters = greedy_clusters(corr, threshold=0.7)
    assert len(clusters) == 1
    members = set(clusters[0]["members"])
    assert members == {"oir", "wdi", "book_slope"}
    assert "depth" in clusters[0]["name"]
    assert clusters[0]["mean_abs_corr"] > 0.9


def test_greedy_clusters_empty_and_singleton():
    corr = {("a", "b"): 0.2, ("a", "c"): float("nan")}
    assert greedy_clusters(corr, threshold=0.7) == []


# --------------------------------------------------------------------- #
# 4. coverage                                                           #
# --------------------------------------------------------------------- #
def test_coverage_full_library():
    from hft_autofactor.ingest import DEFAULT_FACTORS

    cov = coverage_report(list(DEFAULT_FACTORS))
    assert cov["gaps"] == ["time_of_day"]
    assert "iopv" in cov["thin_dimensions"]
    assert cov["dimension_coverage"]["depth"]["covered"] is True
    assert cov["unmapped_factors"] == []
    dims_with_hints = {h["dimension"] for h in cov["opportunity_hints"]}
    assert "time_of_day" in dims_with_hints
    assert "iopv" in dims_with_hints  # thin coverage also gets a hint


def test_coverage_unmapped_and_weakest_horizons():
    decay = decay_table(make_eval_report())
    cov = coverage_report(
        list(FACTORS) + ["my_new_factor_v3"], decay,
        make_eval_report()["stage1_screen"],
    )
    assert cov["unmapped_factors"] == ["my_new_factor_v3"]
    assert set(cov["gaps"]) == {"quote", "iopv", "time_of_day"}
    weakest = cov["weakest_horizons"]
    assert [w["horizon_s"] for w in weakest][:2] == [900, 300]
    assert weakest[0]["stage1_pass"] == 0


# --------------------------------------------------------------------- #
# 5. data quality                                                       #
# --------------------------------------------------------------------- #
def test_panel_quality_rates(digest_root):
    paths = parquet_paths_for_dates(digest_root, ["20250701", "20250702"])
    assert len(paths) == 2
    dq = panel_quality(paths, factor_cols=list(FACTORS))

    n = 800  # 2 partitions x 400
    assert dq["n_rows"] == n
    assert dq["n_partitions"] == 2

    # expected rates from the construction rule (i % 10, i % 7 per part)
    per = np.arange(400)
    exp_one_sided = 2 * (per % 10 == 0).sum() / n
    exp_iopv = 2 * (per % 7 == 0).sum() / n
    exp_clean = 2 * ((per % 10 != 0) & (per % 7 != 0)).sum() / n
    assert dq["flag_bit_rates"]["one_sided_book"] == pytest.approx(exp_one_sided)
    assert dq["flag_bit_rates"]["iopv_invalid"] == pytest.approx(exp_iopv)
    assert dq["flag_bit_rates"]["seq_gap_before"] == 0.0
    assert dq["clean_rows_rate"] == pytest.approx(exp_clean)
    assert dq["one_sided_book_rate"] == pytest.approx(exp_one_sided)
    assert dq["quote_side_missing_rate"] == pytest.approx(exp_one_sided)

    # ABSENT label rates
    assert dq["absent_label_rates"]["fwd_mid_ret_15s"] == pytest.approx(0.10)
    assert dq["absent_label_rates"]["fwd_mid_ret_900s"] == pytest.approx(0.40)
    assert dq["absent_label_rates"]["fwd_last_ret_15s"] == pytest.approx(0.10)

    # NaN rates: cancel everywhere, order_arrival only on SSE
    assert dq["factor_nan_rates"]["cancel_ratio_60s"] == pytest.approx(1.0)
    assert dq["factor_nan_rates"]["order_arrival_60s"] == pytest.approx(0.5)
    assert dq["factor_nan_rates"]["oir"] == pytest.approx(0.0)
    sse = dq["factor_nan_rates_by_exchange"]["sse"]
    szse = dq["factor_nan_rates_by_exchange"]["szse"]
    assert sse["order_arrival_60s"] == pytest.approx(1.0)
    assert szse["order_arrival_60s"] == pytest.approx(0.0)
    assert dq["n_rows_by_exchange"] == {"sse": 400, "szse": 400}


def test_parquet_paths_missing_dates(digest_root):
    paths = parquet_paths_for_dates(digest_root, ["20250701", "20991231"])
    assert len(paths) == 1


# --------------------------------------------------------------------- #
# end-to-end: build_digest + markdown + CLI                             #
# --------------------------------------------------------------------- #
def test_build_digest_end_to_end(digest_root):
    digest = build_digest(digest_root)

    assert digest["panel_available"] is True
    assert digest["dates"] == ["20250701", "20250702"]
    assert digest["n_trials"]["total"] == 3
    assert digest["n_trials"]["by_stage"]["stage1"] == 2
    assert digest["n_trials"]["by_stage"]["stage2_walkforward"] == 1

    # section 1: decay present for every evaluated factor
    assert {r["factor"] for r in digest["ic_decay"]} == set(FACTORS)

    # section 2: taxonomy sees the panel NaN evidence
    tax = {(r["factor"], r["horizon_s"]): r for r in digest["taxonomy"]}
    assert tax[("oir", 60)]["combined_passed"] is True
    assert tax[("order_arrival_60s", 60)]["nan_by_design"] is True
    assert tax[("cancel_ratio_60s", 900)]["failure_reasons"] == ["nan_by_design"]

    # section 3: correlations exclude NaN-by-design columns, cluster family
    corr = digest["correlations"]
    assert "cancel_ratio_60s" not in corr["factors"]
    assert corr["n_sample_rows"] > 0
    names = [c["name"] for c in corr["clusters"]]
    assert any("depth" in nm for nm in names)

    # section 4: coverage gaps include quote/iopv/time_of_day
    assert set(digest["coverage"]["gaps"]) == {"quote", "iopv", "time_of_day"}

    # section 5: data quality filled
    assert digest["data_quality"]["n_rows"] == 800


def test_build_digest_without_panel(digest_root):
    digest = build_digest(digest_root, include_panel=False)
    assert digest["panel_available"] is False
    assert digest["data_quality"]["available"] is False
    assert digest["correlations"]["n_sample_rows"] == 0
    # NaN-by-design still detected via n_obs == 0 fallback
    tax = {(r["factor"], r["horizon_s"]): r for r in digest["taxonomy"]}
    assert tax[("cancel_ratio_60s", 60)]["nan_by_design"] is True


def test_write_digest_json_and_markdown(digest_root, tmp_path):
    digest = build_digest(digest_root)
    json_path, md_path = write_digest(digest, tmp_path / "digest_out")

    assert json_path.name == "digest_20250701_20250702.json"
    assert md_path.name == "digest_20250701_20250702.md"

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert doc["dates"] == ["20250701", "20250702"]

    def _no_nan(obj):
        if isinstance(obj, dict):
            return all(_no_nan(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_no_nan(v) for v in obj)
        if isinstance(obj, float):
            return math.isfinite(obj)
        return True

    assert _no_nan(doc)  # NaN -> null everywhere in the JSON

    md = md_path.read_text(encoding="utf-8")
    for section in ("IC 衰减与半衰期", "通过 / 失败分类", "因子相关性簇",
                    "覆盖缺口与机会提示", "数据质量"):
        assert section in md
    assert "NaN-by-design" in md or "nan" in md.lower()


def test_cli_end_to_end(digest_root, tmp_path):
    report_dir = tmp_path / "digest_cli"
    rc = digest_main([
        "--out-root", str(digest_root),
        "--report-dir", str(report_dir),
    ])
    assert rc == 0
    assert len(list(report_dir.glob("digest_*.json"))) == 1
    assert len(list(report_dir.glob("digest_*.md"))) == 1


def test_cli_dates_spec_and_no_panel(digest_root, tmp_path):
    rc = digest_main([
        "--out-root", str(digest_root),
        "--report-dir", str(tmp_path / "d2"),
        "--dates", "20250701..20250702",
        "--no-panel",
    ])
    assert rc == 0
    doc = json.loads(
        next((tmp_path / "d2").glob("digest_*.json")).read_text(encoding="utf-8")
    )
    assert doc["panel_available"] is False


def test_cli_no_eval_report(tmp_path, capsys):
    out_root = tmp_path / "empty_root"
    out_root.mkdir()
    rc = digest_main(["--out-root", str(out_root)])
    assert rc == 1
    assert "eval" in capsys.readouterr().err


def test_find_eval_report_variants(digest_root, tmp_path):
    # explicit path
    explicit = digest_root / "reports" / "eval_20250701_20250702.json"
    assert find_eval_report(digest_root, explicit=explicit) == explicit
    with pytest.raises(FileNotFoundError):
        find_eval_report(digest_root, explicit=digest_root / "nope.json")
    # by date overlap
    found = find_eval_report(digest_root, dates=["20250701", "20250702"])
    assert found == explicit
    # newest by default
    assert find_eval_report(digest_root) == explicit


def test_ledger_counts(digest_root):
    counts = ledger_counts(digest_root / "reports" / "trial_ledger.jsonl")
    assert counts["total"] == 3
    assert counts["by_stage"] == {"stage1": 2, "stage2_walkforward": 1}
    assert ledger_counts(digest_root / "reports" / "missing.jsonl")["total"] == 0
