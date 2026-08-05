"""Name-uniqueness check for iter-003 specs.

Collects PROTOTYPE names from explore-specs-iter003/*.py and checks against:
  1. the 42 panel columns (base 14 + labels 11 + engine/wishlist 17)
  2. built-in explore prototypes (log_mid_ret_60s, spread_z_300s, depth_imbalance_5l)
  3. the 18 registered prototypes from iter-001/002
  4. duplicates within the batch
Exit code 0 = clean, 1 = collisions found.
"""
import ast
import sys
from pathlib import Path

SPEC_DIR = Path(r"D:\claude\Quant_works\hft-autofactor\explore-specs-iter003")

PANEL_COLS = {
    # base 14
    "date", "exchange", "instrument", "ts_ms", "snap_seq", "flags",
    "mid_px", "last_px", "bid1_px", "ask1_px", "bid1_qty", "ask1_qty",
    "depth_bid5", "depth_ask5", "channel",
    # engine 12
    "quoted_spread_ticks", "microprice_dev", "oir", "wdi", "book_slope",
    "iopv_premium", "rv_60s", "rv_300s", "ofi_60s", "trade_imbalance_60s",
    "order_arrival_60s", "cancel_ratio_60s",
    # wishlist 5
    "avg_trade_size_60s", "n_trades_60s", "large_trade_share_60s",
    "trade_gap_ms", "cum_trade_vol",
    # labels 11
    *(f"fwd_{m}_ret_{h}s" for m in ("mid", "last") for h in (15, 30, 60, 300, 900)),
    "log_mid_ret_60s",
}
BUILTIN_PROTOS = {"log_mid_ret_60s", "spread_z_300s", "depth_imbalance_5l"}
REGISTERED_18 = {
    "dd_flow_300s", "depth5_delta_60s", "depth_resiliency",
    "flow_divergence_300s", "iopv_premium_mom", "large_trade_share_level",
    "ofi_accum_300s", "prem_x_ofi", "prem_x_wdi", "queue_refill_asym_300s",
    "rv_asym_300s", "session_clock", "ti_accum_300s", "ti_ewm_accel_120s",
    "ti_ewm_state_300s", "trade_arrival_burst", "vol_adj_slope", "wdi_mom_90s",
}


def extract_name(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "PROTOTYPE":
                    call = node.value
                    if isinstance(call, ast.Call) and call.keywords:
                        for kw in call.keywords:
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                                return kw.value.value
    return None


def main():
    specs = sorted(SPEC_DIR.glob("*.py"))
    print(f"spec files: {len(specs)}")
    names = {}
    missing = []
    for p in specs:
        n = extract_name(p)
        if n is None:
            missing.append(p.name)
        else:
            names[p.name] = n
    if missing:
        print("NO PROTOTYPE name found:", missing)

    problems = []
    for fname, n in names.items():
        if n in PANEL_COLS:
            problems.append(f"{fname}: name '{n}' collides with a panel column")
        if n in BUILTIN_PROTOS:
            problems.append(f"{fname}: name '{n}' collides with built-in prototype")
        if n in REGISTERED_18:
            problems.append(f"{fname}: name '{n}' collides with registered prototype")

    seen = {}
    for fname, n in names.items():
        if n in seen:
            problems.append(f"in-batch duplicate: '{n}' in {seen[n]} and {fname}")
        seen[n] = fname

    print(f"unique prototype names: {len(seen)}")
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    print("CLEAN: no collisions")
    for n in sorted(seen):
        print(" ", n)


if __name__ == "__main__":
    main()
