"""Smoke harness for explore-lane prototype specs (iter-003 batch protocol).

Per spec file checks:
  1. loads via load_prototype_spec (metadata completeness + name rules)
  2. prototype name == file stem
  3. compute(synthetic day-part) runs and returns an aligned column
  4. output finite; non-null fraction in the tail; not constant
  5. CAUSALITY PROBE: perturbing the last third of the rows must not change
     any output in the first two thirds

Usage:
  /d/claude/Quant_works/venv/Scripts/python smoke_specs.py <files or dirs...>

Exit code 1 if any spec fails. Local-only: never touches the server.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from hft_autofactor.explore.registry import PrototypeError, load_prototype_spec

N = 1500          # rows (~75 min of 3s snapshots; covers 600-row warm-ups)
PROBE_CUT = 1000  # perturb rows >= cut; outputs < cut must be identical


def _causal_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.full(x.shape, np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def synth_arrays(n: int = N, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    tick = 0.001
    ts_ms = 34_200_000 + np.arange(n) * 3000
    steps = rng.normal(0.0, 0.8, n)
    mid = (1200.0 + np.round(np.cumsum(steps))) * tick
    spread_t = rng.choice([1.0, 1.0, 2.0], n)
    bid1 = mid - spread_t * tick / 2.0
    ask1 = mid + spread_t * tick / 2.0
    u = rng.random(n)
    last = np.where(u < 0.45, bid1, np.where(u < 0.9, ask1, mid))
    bid1_qty = np.round(np.exp(rng.normal(7.5, 1.0, n)))
    ask1_qty = np.round(np.exp(rng.normal(7.5, 1.0, n)))
    walk = np.tanh(np.cumsum(rng.normal(0.0, 0.05, n)))
    return {
        "date": ["20250701"] * n,
        "exchange": ["SSE"] * n,
        "instrument": ["588000"] * n,
        "ts_ms": ts_ms,
        "snap_seq": np.arange(n),
        "flags": np.zeros(n),
        "mid_px": mid,
        "last_px": last,
        "bid1_px": bid1,
        "ask1_px": ask1,
        "bid1_qty": bid1_qty,
        "ask1_qty": ask1_qty,
        "depth_bid5": bid1_qty * rng.uniform(2.0, 6.0, n),
        "depth_ask5": ask1_qty * rng.uniform(2.0, 6.0, n),
        "channel": ["SSE"] * n,
        "quoted_spread_ticks": spread_t,
        "microprice_dev": rng.normal(0.0, 0.0002, n),
        "oir": walk,
        "wdi": np.tanh(np.cumsum(rng.normal(0.0, 0.04, n))),
        "book_slope": _causal_mean(rng.normal(0.0, 1.0, n), 10),
        "iopv_premium": np.clip(np.cumsum(rng.normal(0.0, 8e-5, n)), -0.005, 0.005),
        "ofi_60s": _causal_mean(rng.normal(0.0, 400.0, n), 20),
        "trade_imbalance_60s": np.tanh(np.cumsum(rng.normal(0.0, 0.06, n))),
        "order_arrival_60s": np.full(n, np.nan),  # SSE: never available
        "cancel_ratio_60s": np.clip(_causal_mean(rng.uniform(0.0, 1.0, n), 20), 0, 1),
        "avg_trade_size_60s": np.exp(_causal_mean(rng.normal(6.5, 0.5, n), 20)),
        "n_trades_60s": np.nan_to_num(_causal_mean(rng.poisson(12, n).astype(float), 10)),
        "large_trade_share_60s": np.clip(_causal_mean(rng.normal(0.3, 0.2, n), 20), 0, 1),
        "trade_gap_ms": rng.exponential(1200.0, n) + 40.0,
        "cum_trade_vol": np.cumsum(np.exp(rng.normal(4.0, 0.8, n))),
        # --- batch-2 wishlist columns (#144, materialized 2026-08-06) ---
        "total_bid_vol": np.exp(rng.normal(9.5, 0.5, n)),
        "total_ask_vol": np.exp(rng.normal(9.4, 0.5, n)),
        "bid_orders5": np.round(np.exp(rng.normal(2.2, 0.6, n))),
        "ask_orders5": np.round(np.exp(rng.normal(2.1, 0.6, n))),
        "open_px": np.full(n, mid[0]),
        "high_px": np.maximum.accumulate(last),
        "low_px": np.minimum.accumulate(last),
        "pre_close_px": np.full(n, mid[0] * 0.998),
        "iopv_velocity": rng.normal(0.0, 0.03, n),
        "ofi_15s": _causal_mean(rng.normal(0.0, 400.0, n), 5),
        "ofi_30s": _causal_mean(rng.normal(0.0, 400.0, n), 10),
        "trade_imbalance_15s": np.tanh(np.cumsum(rng.normal(0.0, 0.10, n))),
        "trade_imbalance_30s": np.tanh(np.cumsum(rng.normal(0.0, 0.08, n))),
        "buy_vol_60s": np.exp(_causal_mean(rng.normal(6.5, 0.6, n), 20)),
        "sell_vol_60s": np.exp(_causal_mean(rng.normal(6.4, 0.6, n), 20)),
        "large_trade_net_share_60s": np.clip(_causal_mean(rng.normal(0.0, 0.5, n), 20), -1, 1),
        "book_event_intensity_60s": rng.exponential(6.0, n) + 2.0,
    }


def build_df(arrays: dict) -> pl.DataFrame:
    df = pl.DataFrame(arrays)
    ret = df["mid_px"].log().diff()
    df = df.with_columns(
        (ret.pow(2).rolling_mean(window_size=20, min_samples=20) * 20).alias("rv_60s"),
        (ret.pow(2).rolling_mean(window_size=100, min_samples=100) * 100).alias("rv_300s"),
    )
    return df


def perturbed(arrays: dict, cut: int = PROBE_CUT) -> dict:
    out = {}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            v2 = v.astype(float).copy()
            v2[cut:] = v2[cut:] + 1.0 + np.abs(v2[cut:]) * 0.01
            out[k] = v2
        else:
            out[k] = v
    return out


def to_numpy(out) -> np.ndarray:
    if isinstance(out, pl.Series):
        return out.to_numpy()
    if isinstance(out, pl.DataFrame):
        if out.width != 1:
            raise ValueError(f"compute returned DataFrame with {out.width} columns")
        return out.to_series().to_numpy()
    arr = np.asarray(out)
    if arr.ndim != 1:
        raise ValueError(f"compute returned array with shape {arr.shape}")
    return arr


def check_spec(path: Path, part_a: pl.DataFrame, part_b: pl.DataFrame) -> tuple[bool, str, str]:
    try:
        proto = load_prototype_spec(path)
    except PrototypeError as e:
        return False, "", f"registration refused: {e}"
    except Exception as e:
        return False, "", f"import failed: {type(e).__name__}: {e}"

    if proto.name != path.stem:
        return False, proto.name, f"name {proto.name!r} != file stem {path.stem!r}"

    try:
        out_a = to_numpy(proto.compute(part_a))
        out_b = to_numpy(proto.compute(part_b))
    except Exception as e:
        return False, proto.name, f"compute failed: {type(e).__name__}: {e}"

    if len(out_a) != N or len(out_b) != N:
        return False, proto.name, f"output length {len(out_a)}/{len(out_b)} != {N}"

    a = out_a.astype(float)
    b = out_b.astype(float)
    if np.isinf(a[~np.isnan(a)]).any():
        return False, proto.name, "output contains inf"

    prefix_a, prefix_b = a[:PROBE_CUT], b[:PROBE_CUT]
    same = np.isclose(prefix_a, prefix_b, rtol=0.0, atol=1e-12, equal_nan=True)
    if not same.all():
        bad = int(np.argmax(~same))
        return False, proto.name, (
            f"CAUSALITY VIOLATION: perturbing rows >= {PROBE_CUT} changed output "
            f"at row {bad} ({prefix_a[bad]!r} -> {prefix_b[bad]!r})"
        )

    tail = a[PROBE_CUT:]
    nonnull = float(np.mean(~np.isnan(tail)))
    if nonnull < 0.3:
        return False, proto.name, f"tail non-null fraction {nonnull:.2f} < 0.30"
    finite_tail = tail[~np.isnan(tail)]
    if finite_tail.size > 0 and float(np.std(finite_tail)) == 0.0:
        return False, proto.name, "output is constant (no rank information)"

    return True, proto.name, f"ok (tail non-null {nonnull:.2f})"


def main(argv: list[str]) -> int:
    paths: list[Path] = []
    for arg in argv:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.py")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"skip (not found): {arg}")
    if not paths:
        print("no spec files given")
        return 2

    base = synth_arrays()
    part_a = build_df(base)
    part_b = build_df(perturbed(base))

    n_ok = 0
    seen: dict[str, Path] = {}
    for path in paths:
        ok, name, msg = check_spec(path, part_a, part_b)
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {path.name}: {msg}")
        if ok:
            n_ok += 1
            if name in seen:
                print(f"[FAIL] duplicate name {name!r}: {seen[name].name} and {path.name}")
                n_ok -= 1
            else:
                seen[name] = path
    print(f"\n{n_ok}/{len(paths)} specs passed")
    return 0 if n_ok == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
