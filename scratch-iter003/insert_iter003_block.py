"""Insert iter-003 round-1 results block into library/candidates.json.

Builds the block programmatically from round1_admitted_detail.json +
hand-written lessons/fail-groups, then text-splices it as a new top-level
key after eval_v2_rescreen_2026_08_05. Dual validation (block + full file)
before writing. LF line endings preserved.
"""
import json
from pathlib import Path

ROOT = Path(r"D:\claude\Quant_works\hft-autofactor")
CAND = ROOT / "library" / "candidates.json"
DETAIL = ROOT / "scratch-iter003" / "round1_admitted_detail.json"

detail = json.loads(DETAIL.read_text(encoding="utf-8"))
OLD_ADMITTED_N = len(json.loads(CAND.read_text(encoding="utf-8"))["admitted"])

block_obj = {
    "iter003_round1": {
        "trigger": ("iter-003 first batch: 60 wide-table-derived candidates "
                    "(families A depth/book, B flow interaction, C price/vol, "
                    "D etf regime) registered + run + eval-v2 screen + "
                    "within-batch Spearman dedup"),
        "generated_at": "2026-08-05",
        "batch": {
            "specs_dir": "explore-specs-iter003",
            "n_specs": 60,
            "families": {
                "A_depth_book": 9, "B_flow_interaction": 16,
                "C_price_vol": 19, "D_etf_regime": 16,
            },
        },
        "gate": {
            "version": "eval_v2",
            "is_t_min": 2.0, "oos_t_min": 2.0,
            "retention_min": 0.5,
            "retention_sign_rule": "IS/OOS IC must have same sign; flip fails",
            "dedup_max_abs_corr_vs_panel": 0.85,
            "dedup_library": "17 wide-table factor columns",
            "embargo_days": 1, "n_test_days": 5,
            "cost_bps_per_side": 3.0,
            "head_taus": [0.01, 0.05, 0.10],
            "head_stats_role": "descriptive only, not a gate",
        },
        "split": {
            "train": "20250701-20250922 (60d)",
            "test": "20250924-20250930 (5d)",
            "embargo": "20250923",
            "panel_rows": 314053, "panel_days": 66,
        },
        "pipeline_run": {
            "registry": "81 prototypes in prod registry (18 prior + 3 built-in + 60 new)",
            "run": "60 ok / 0 failed (~2.5 min)",
            "screen": "19 pass / 41 fail (~3 min)",
        },
        "summary": {
            "screen_pass": 19,
            "screen_fail": 41,
            "dedup_killed_within_batch": 3,
            "admitted_new": 16,
            "prior_admitted_still_valid": 6,
            "library_total": 22,
        },
        "dedup_within_batch": {
            "method": ("pooled Spearman over 314,053 rows x 66 days among the "
                       "19 screen-pass protos + 6 prior admitted (25 columns, "
                       "300 pairs); threshold 0.85"),
            "high_pairs": [
                {"a": "microprice_dev_mom_60s", "b": "oir_mom_60s", "rho": 0.996,
                 "kept": "oir_mom_60s",
                 "reason": "IC identical to 4 decimals; OIR momentum is the more direct mechanism"},
                {"a": "signed_rv_60s", "b": "vol_adj_mom_60s", "rho": 0.974,
                 "kept": "signed_rv_60s",
                 "reason": ("vol_adj_mom_60s sits at 0.849 vs library ofi_60s "
                            "(near re-skin); signed_rv_60s more independent (0.782)")},
                {"a": "top_book_delta_30s", "b": "wdi_mom_30s", "rho": 0.878,
                 "kept": "wdi_mom_30s",
                 "reason": ("lower panel corr 0.558 vs 0.622, higher OOS IC; "
                            "top-book family keeps top_book_delta_120s (rho 0.454)")},
            ],
            "notable_subthreshold": [
                {"a": "flow_divergence_120s", "b": "flow_divergence_300s (prior)",
                 "rho": 0.783, "note": "family passes at 60/120/300s windows; saturated, do not extend"},
                {"a": "flow_divergence_60s", "b": "flow_divergence_120s", "rho": 0.763},
                {"a": "signed_rv_60s", "b": "ofi_60s (panel)", "rho": 0.782},
                {"a": "log_mid_ret_120s", "b": "ofi_60s (panel)", "rho": 0.616},
            ],
            "artifacts": ["iter003_round1_corr.json", "iter003_round1_corr.txt"],
        },
        "admitted": detail["admitted"],
        "dedup_killed": detail["dedup_killed"],
        "fail_groups": {
            "premium_iopv": {
                "n": 8,
                "names": ["iopv_premium_mom_60s", "iopv_premium_z_120s",
                          "iopv_premium_z_600s", "premium_dev_day",
                          "prem_reversion_x_rv", "prem_x_cancel",
                          "prem_x_depth_imb", "ofi_x_premium_sign"],
                "cause": ("regime break: IS t -3..-8 collapses to OOS |t|<=1.5, "
                          "retention <=0.31; confirms rescreen warning - "
                          "unconditional premium factors are dead"),
            },
            "rv_vol": {
                "n": 5,
                "names": ["rv_z_300s", "rv_ratio_60_300", "rv_ratio_z_300s",
                          "vol_confirmed_mom", "vol_rate_x_ti"],
                "cause": "RV z/ratio variants IS-dead (|t|<=2); vol_rate_x_ti IS t 5.7 but OOS retention 0.27",
            },
            "ofi_flow": {
                "n": 5,
                "names": ["ofi_fast_slow", "ofi_mom_60s", "ofi_ti_agree_60s",
                          "cancel_x_ofi", "flow_accel_signed"],
                "cause": ("IS t 5-10 collapses to OOS |t|<=1.7 (retention 0.5-0.64 "
                          "just short); flow_accel_signed flips sign; cancel_x_ofi "
                          "structurally NaN on SSE"),
            },
            "ti_aggressor": {
                "n": 5,
                "names": ["ti_accum_120s", "ti_z_x_large_share", "ti_z_x_spread_z",
                          "arrival_accel_x_ti", "aggressor_share_60s"],
                "cause": "retention collapse 0.10-0.20 or IS-dead; aggressor_share_60s OOS t 1.61 just short @900s",
            },
            "depth_slope_level": {
                "n": 5,
                "names": ["book_slope_delta_60s", "book_slope_z_300s",
                          "depth_ratio_5to1_z", "depth_thickness_z_300s",
                          "queue_pressure_x_slope"],
                "cause": "IS-dead or sign-flip; queue_pressure_x_slope IS t 7.2 but retention 0.35",
            },
            "short_momentum": {
                "n": 5,
                "names": ["log_mid_ret_15s", "log_mid_ret_30s", "last_mid_gap_ma_30s",
                          "gap_x_direction", "signed_arrival_z"],
                "cause": ("raw short-window momentum fails retention (0.33-0.49) or "
                          "sign-flips; only log_mid_ret_120s survives (at 900s, admitted); "
                          "signed_arrival_z uses order_arrival_60s = NaN on SSE"),
            },
            "spread": {
                "n": 2,
                "names": ["spread_z_60s", "spread_z_120s"],
                "cause": ("spread level IS-dead (|t|<=1.4): spread alone has no "
                          "directional power; it works only as a conditioning state "
                          "(ofi_z_x_spread_z / flow_divergence_x_spread_z pass)"),
            },
            "large_trade": {
                "n": 2,
                "names": ["large_share_mom_300s", "large_share_z_x_ti"],
                "cause": "IS-dead (|t|<=1.5)",
            },
            "session_regime": {
                "n": 2,
                "names": ["session_u_x_mom", "regime_vol_x_flow"],
                "cause": "OOS collapse (retention 0.08-0.43)",
            },
            "duplicate_of_library": {
                "n": 1,
                "names": ["microprice_dev_z_300s"],
                "cause": "screen dedup gate: 0.902 vs library microprice_dev (IC itself passed 15/30s)",
            },
            "other": {
                "n": 1,
                "names": ["size_x_direction"],
                "cause": "sign-flip IS->OOS",
            },
        },
        "lessons": [
            "Book-imbalance momentum/delta is the strongest 15/30s cluster (8 of 16 admitted): OOS IC 0.09-0.17, t 5-15; wdi_accel_90s spans 15/30/60/900s.",
            "Within-family redundancy is severe: 3 of 19 screen-pass protos are within-batch duplicates (rho 0.88-0.996); same construction on different bases (oir vs microprice_dev momentum) can be near-identical. Check within-family correlations BEFORE submitting more variants.",
            "mid_day_range_pos passes ALL 5 horizons (negative IC: near day-high => lower future return, mean reversion) and is independent of everything (max rho 0.41) - a genuinely new signal source; round 2 should expand it (range expansion speed, proximity-to-low variants).",
            "flow_divergence passes at 60/120/300s windows (pairwise rho 0.55-0.78): robust family but saturated; do not extend further.",
            "Premium/iopv family 0/8 (regime break confirmed twice now). Only regime-CONDITIONAL premium constructions are worth trying; no more unconditional premium factors.",
            "cancel_ratio_60s and order_arrival_60s are UNAVAILABLE on SSE (all NaN): anything multiplying them (cancel_x_ofi, prem_x_cancel, signed_arrival_z) is structurally dead. Never use in SSE specs.",
            "Raw short-window momentum (log_mid_ret_15s/30s) fails retention; momentum needs book/flow information to work at short horizons. 300/900s winners are slow state variables (depth5_delta_120s, mid_day_range_pos, log_mid_ret_120s, price_accel_60_180, signed_rv_60s).",
            "Spread level has no direct predictive power (spread_z IS-dead) but flow-x-spread-state interactions pass @15s (ofi_z_x_spread_z, flow_divergence_x_spread_z): condition flow by spread state.",
            "Head stats: tau5% head gross returns mostly 1-6bp, cannot cover 6bp round-trip; net mostly negative. Value is in ranking/conditioning, head stats stay descriptive only.",
        ],
        "artifacts": {
            "screen_reports": "scratch-iter003/reports_round1/ (60 JSONs)",
            "screen_log": "scratch-iter003/iter003_round1_screen.log",
            "corr": ("scratch-iter003/iter003_round1_corr.json + .txt "
                     "(server copies: /data/factor_lzt/iterations/)"),
            "admitted_detail": "scratch-iter003/round1_admitted_detail.json",
            "chinese_report": ("scratch-iter003/iter003_round1_report.md "
                               "(server copy: /data/factor_lzt/iterations/)"),
        },
    }
}

# --- build block text with file-matching indentation ---
wrapped = json.dumps(block_obj, indent=2, ensure_ascii=False)
assert wrapped.startswith("{\n") and wrapped.endswith("\n}")
block = wrapped[2:-2].rstrip("\n")          # strip outer braces
# validate block in isolation
json.loads("{\n" + block + "\n}")

# --- splice ---
s = CAND.read_text(encoding="utf-8")
anchor = "\n  }\n}"
assert s.endswith(anchor), f"unexpected tail: {s[-40:]!r}"
new = s[:-len(anchor)] + "\n  },\n" + block + "\n}"
full = json.loads(new)                       # validate full file
assert "iter003_round1" in full
assert len(full["admitted"]) == OLD_ADMITTED_N, "old admitted list must be untouched"

CAND.write_text(new, encoding="utf-8", newline="")
print("OK: candidates.json updated")
print("  top-level keys:", list(full.keys()))
print("  admitted new:", len(full["iter003_round1"]["admitted"]))
print("  file chars:", len(new))
