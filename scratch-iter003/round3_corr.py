"""iter-003 round 3: pooled Spearman dedup among PASS_NEW (argv) + 35 library.

Usage: python round3_corr.py <pass1> <pass2> ...
Loads explore panels from /data/factor_lzt/prod/explore/panels/{name}/,
joins on (date, ts_ms), computes pairwise Spearman over pooled rows.
Prints all pairs with |rho| > 0.50, marks >= 0.85 HIGH. Dumps JSON.
Run ON the server (conda env autofactor).
"""
import json
import sys
from pathlib import Path

import polars as pl

PANELS = Path("/data/factor_lzt/prod/explore/panels")
OUT_JSON = Path("/data/factor_lzt/iterations/iter003_round3_corr.json")
OUT_TXT = Path("/data/factor_lzt/iterations/iter003_round3_corr.txt")

# 35 admitted library factors (16 round-1 + 6 eval-v2 rescreen + 13 round-2)
LIBRARY = [
    # round 1 (16)
    "oir_mom_60s", "top_book_delta_120s", "wdi_mom_30s", "wdi_mom_180s",
    "depth5_delta_30s", "wdi_accel_90s", "last_mid_gap_ticks",
    "mid_day_range_pos", "ofi_z_x_spread_z", "flow_divergence_120s",
    "flow_divergence_60s", "flow_divergence_x_spread_z", "depth5_delta_120s",
    "log_mid_ret_120s", "price_accel_60_180", "signed_rv_60s",
    # eval-v2 rescreen (6)
    "flow_divergence_300s", "depth5_delta_60s", "wdi_mom_90s",
    "dd_flow_300s", "rv_asym_300s", "session_clock",
    # round 2 (13)
    "dev_from_open_bps", "mid_roll_range_pos_300s", "range_pos_x_spread_z",
    "ti_accel_15_60", "ofi_15s_z_120s", "ofi_concord_15_60",
    "ofi_per_depth_z_300s", "fullbook_imb_mom_60s", "fullbook_imb_z_300s",
    "conc_imb_z_300s", "top5_book_div_z_300s", "iopv_vel_z_300s",
    "iopv_vel_drift_300s",
]


def load(name: str) -> pl.DataFrame:
    d = PANELS / name
    files = sorted(d.glob("dt=*.parquet"))
    if not files:
        raise SystemExit(f"no panels for {name} in {d}")
    parts = []
    for f in files:
        df = pl.read_parquet(f)
        cols = df.columns
        if name not in cols:
            raise SystemExit(f"column {name} missing in {f}: {cols}")
        if "date" not in cols:
            dt = f.parent.name.split("=")[-1] if f.parent.name.startswith("dt=") else f.stem.split("=")[-1]
            df = df.with_columns(pl.lit(dt).alias("date"))
        if "ts_ms" not in df.columns:
            raise SystemExit(f"ts_ms missing in {f}: {df.columns}")
        parts.append(df.select(["date", "ts_ms", name]).with_columns(
            pl.col("date").cast(pl.Utf8), pl.col("ts_ms").cast(pl.Int64)
        ))
    return pl.concat(parts, how="vertical")


def spearman(x, y) -> float:
    """Spearman rho = Pearson on ranks, via primitive ops only.

    polars 1.43 removed the Series-level corr methods (pearson_corr, corr);
    rank/mean/sum arithmetic is API-stable, so compute it by hand.
    """
    rx = x.rank()
    ry = y.rank()
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    num = float((dx * dy).sum())
    den = (float((dx * dx).sum()) * float((dy * dy).sum())) ** 0.5
    if den <= 0.0:
        return float("nan")
    return num / den


def main() -> None:
    pass_new = sys.argv[1:]
    if not pass_new:
        raise SystemExit("usage: round3_corr.py <pass1> <pass2> ...")
    names = pass_new + LIBRARY
    wide = None
    for n in names:
        df = load(n)
        if wide is None:
            wide = df
        else:
            wide = wide.join(df, on=["date", "ts_ms"], how="inner")
        print(f"loaded {n}: wide rows={wide.height}", flush=True)

    print(f"pooled wide: {wide.height} rows x {wide.width} cols", flush=True)
    wide = wide.sort(["date", "ts_ms"])

    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sub = wide.select([a, b]).drop_nulls()
            n_obs = sub.height
            if n_obs < 1000:
                rho = float("nan")
            else:
                rho = spearman(sub[a], sub[b])
            pairs.append({"a": a, "b": b, "rho": rho, "n": n_obs})

    pairs.sort(key=lambda p: -abs(p["rho"]) if p["rho"] == p["rho"] else 0)
    OUT_JSON.write_text(json.dumps({
        "n_rows_pooled": wide.height,
        "n_dates": int(wide.select("date").n_unique()),
        "names": names,
        "pairs": pairs,
    }, indent=1), encoding="utf-8")

    lines = [f"pooled rows={wide.height} dates={wide.select('date').n_unique()}"]
    for p in pairs:
        if p["rho"] != p["rho"] or abs(p["rho"]) < 0.50:
            continue
        tag = "  <<< HIGH" if abs(p["rho"]) >= 0.85 else ""
        lines.append(
            f"{p['a']:32s} x {p['b']:32s} n={p['n']:7d} rho={p['rho']:+.3f}{tag}"
        )
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    sys.exit(main())
