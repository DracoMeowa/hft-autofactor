"""Multiple-testing gates for the factor zoo (Stage 4 screening).

Every mined variant is a trial; with an auto-mining zoo the honest trial
count N drives every threshold:

* t-hurdle:            max(3.0, sqrt(2 ln N_eff))   (Harvey-Liu-Zhu 2016)
* BHY FDR:             Benjamini-Yekutieli at q <= 0.10 (arbitrary dependence)
* Deflated Sharpe:     Bailey & Lopez de Prado (2014), p <= 0.05 <=> DSR >= 0.95
* permutation floor:   99.9th pct of label-shuffled null ICs through the
                       identical pipeline
* trial ledger:        append-only JSONL; every evaluation is logged BEFORE
                       thresholds are read, so N can never be understated

Stage 2 then applies the pristine-OOS gate: retention >= 0.5 with sign
consistency, OOS t >= 2.0 and a minimum win rate.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from .ic import ICStats, label_column, rank_ic_time_series

EULER_GAMMA = 0.5772156649015329

#: minimum OOS IC win rate (fraction of day/instrument ICs with the same
#: sign as the mean) required by the Stage-2 gate
MIN_OOS_WIN_RATE = 0.55


@dataclass
class GateConfig:
    min_rank_ic: dict[int, float] = field(
        default_factory=lambda: {15: 0.02, 30: 0.02, 60: 0.02, 300: 0.03, 900: 0.03}
    )
    min_icir: float = 0.5
    fdr_q: float = 0.10
    min_oos_t: float = 2.0
    min_retention: float = 0.5
    noise_floor_pct: float = 99.9


# --------------------------------------------------------------------- #
# normal distribution helpers (no scipy)                                #
# --------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, |err| < 1.2e-9)."""
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")
        return float("nan")
    a = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q)
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


# --------------------------------------------------------------------- #
# hurdles                                                               #
# --------------------------------------------------------------------- #
def t_hurdle(n_trials_eff: int) -> float:
    """max(3.0, sqrt(2 ln N)): expected max t-stat of N independent trials."""
    n = max(2, int(n_trials_eff))
    return max(3.0, math.sqrt(2.0 * math.log(n)))


def bhy_critical_values(m: int, q: float = 0.10) -> np.ndarray:
    """Benjamini-Yekutieli critical values k/(m*c(m))*q, k=1..m.

    Valid under ARBITRARY dependence (mined factors are correlated), at the
    price of the harmonic factor c(m) ~= ln(m) + 0.577.
    """
    if m < 1:
        return np.zeros(0)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    k = np.arange(1, m + 1, dtype=np.float64)
    return k / (m * c_m) * q


def _bhy_pass(p_values: Sequence[float], q: float) -> list[bool]:
    """Step-up BHY decision for a batch of p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    crit = bhy_critical_values(m, q)
    order = np.argsort(p, kind="mergesort")
    passed = np.zeros(m, dtype=bool)
    best_k = -1
    for rank, idx in enumerate(order):  # rank 0-based -> k = rank+1
        if p[idx] <= crit[rank]:
            best_k = rank
    if best_k >= 0:
        passed[order[: best_k + 1]] = True
    return passed.tolist()


def deflated_sharpe_pvalue(
    sr: float, n_trials: int, T: int, skew: float, kurt: float
) -> float:
    """p-value that the observed SR (or ICIR proxy) is luck (Bailey-LdP DSR).

    DSR = 1 - p; the admission gate requires p <= 0.05 (DSR >= 0.95).
    ``kurt`` is the kurtosis as used in
    sigma_SR^2 = (1 - skew*SR + (kurt-1)*SR^2/4) / (T-1).
    """
    if T <= 2 or not math.isfinite(sr):
        return 1.0
    var_sr = (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr) / (T - 1.0)
    if var_sr <= 0.0 or not math.isfinite(var_sr):
        return 1.0
    sigma_sr = math.sqrt(var_sr)
    n = max(1, int(n_trials))
    if n <= 1:
        sr0 = 0.0
    else:
        z1 = norm_ppf(1.0 - 1.0 / n)
        z2 = norm_ppf(1.0 - 1.0 / (n * math.e))
        sr0 = sigma_sr * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    p = 1.0 - norm_cdf((sr - sr0) / sigma_sr)
    return float(min(1.0, max(0.0, p)))


# --------------------------------------------------------------------- #
# permutation noise floor                                               #
# --------------------------------------------------------------------- #
def permutation_noise_floor(
    panel: pl.DataFrame,
    factor: str,
    horizon_s: int,
    *,
    n_perms: int = 50,
    seed: int = 0,
    label: str = "fwd_mid_ret",
) -> float:
    """Pct-99.9 (default) of mean|IC| under within-group label shuffling.

    Labels are permuted inside each (date, instrument) block so the marginal
    distributions and day structure are preserved while the timestamp
    alignment -- and thus any true signal -- is destroyed.  The returned
    value is the noise floor a real factor's |mean IC| must exceed.
    """
    col_l = label_column(horizon_s, label)
    for col in ("date", "instrument", factor, col_l):
        if col not in panel.columns:
            raise KeyError(f"panel lacks column {col!r}")
    base = (
        panel.select(["date", "instrument", factor, col_l])
        .drop_nulls(subset=[factor, col_l])
    )
    if base.is_empty():
        return float("nan")

    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(max(1, n_perms)):
        shuffled_parts: list[pl.DataFrame] = []
        for part in base.partition_by(["date", "instrument"]):
            y = part[col_l].to_numpy().copy()
            rng.shuffle(y)
            shuffled_parts.append(part.with_columns(pl.Series(col_l, y)))
        perm_panel = pl.concat(shuffled_parts)
        ic_df = rank_ic_time_series(perm_panel, factor, horizon_s, label=label)
        arr = ic_df["ic"].to_numpy().astype(np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            stats.append(float(np.abs(arr).mean()))
    if not stats:
        return float("nan")
    return float(np.percentile(np.asarray(stats), 99.9))


# --------------------------------------------------------------------- #
# trial ledger                                                          #
# --------------------------------------------------------------------- #
class TrialLedger:
    """Append-only JSONL ledger of every evaluated variant (honest N).

    Every multiple-testing correction is only as good as the declared N;
    the ledger is written BEFORE thresholds are read so N can never be
    understated after the fact.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        factor: str,
        horizon_s: int,
        params: dict,
        stage: str,
        metrics: dict,
    ) -> None:
        entry = {
            "ts": time.time(),
            "factor": factor,
            "horizon_s": int(horizon_s),
            "params": params,
            "stage": stage,
            "metrics": metrics,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def n_trials(self, stage: str | None = None) -> int:
        if not self.path.is_file():
            return 0
        count = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if stage is None:
                    count += 1
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("stage") == stage:
                    count += 1
        return count


# --------------------------------------------------------------------- #
# Stage 1: IS screen                                                    #
# --------------------------------------------------------------------- #
def _p_value_two_sided(t: float) -> float:
    if not math.isfinite(t):
        return 1.0
    return float(min(1.0, 2.0 * (1.0 - norm_cdf(abs(t)))))


def stage1_screen(
    stats: Sequence[ICStats],
    ledger: TrialLedger,
    cfg: GateConfig,
    noise_floors: dict[tuple[str, int], float],
) -> pl.DataFrame:
    """Screen candidate (factor, horizon) ICStats through the Stage-1 gates.

    Gates: |mean IC| >= per-horizon floor; |ICIR| >= min_icir; NW t >=
    max(3, sqrt(2 ln N)) with N from the ledger; |mean IC| above the
    permutation noise floor (when provided); BHY-FDR q <= fdr_q over the
    batch.  Every candidate is appended to the ledger at stage="stage1"
    BEFORE the hurdle is read.
    """
    if not stats:
        return pl.DataFrame(
            schema={
                "factor": pl.Utf8, "horizon_s": pl.Int32, "n_obs": pl.UInt32,
                "mean_ic": pl.Float64, "icir": pl.Float64, "t_stat_nw": pl.Float64,
                "n_trials": pl.UInt32, "t_hurdle_min": pl.Float64,
                "noise_floor": pl.Float64, "p_value": pl.Float64,
                "fdr_pass": pl.Boolean, "passed": pl.Boolean,
            }
        )

    for s in stats:
        ledger.log(
            factor=s.factor,
            horizon_s=s.horizon_s,
            params={},
            stage="stage1",
            metrics={
                "mean_ic": s.mean_ic,
                "icir": s.icir,
                "t_stat_nw": s.t_stat_nw,
                "n_obs": s.n_obs,
            },
        )

    n_trials = max(2, ledger.n_trials())
    hurdle = t_hurdle(n_trials)
    p_values = [_p_value_two_sided(s.t_stat_nw) for s in stats]
    fdr_pass = _bhy_pass(p_values, cfg.fdr_q)

    rows: list[dict[str, Any]] = []
    for s, p, fp in zip(stats, p_values, fdr_pass):
        min_ic = cfg.min_rank_ic.get(s.horizon_s, 0.02)
        floor = noise_floors.get((s.factor, s.horizon_s), float("nan"))
        abs_mean = abs(s.mean_ic) if math.isfinite(s.mean_ic) else 0.0
        abs_icir = abs(s.icir) if math.isfinite(s.icir) else 0.0
        above_noise = not math.isfinite(floor) or abs_mean >= floor
        passed = bool(
            abs_mean >= min_ic
            and abs_icir >= cfg.min_icir
            and math.isfinite(s.t_stat_nw)
            and s.t_stat_nw >= hurdle
            and above_noise
            and fp
        )
        rows.append(
            {
                "factor": s.factor,
                "horizon_s": s.horizon_s,
                "n_obs": s.n_obs,
                "mean_ic": s.mean_ic,
                "icir": s.icir,
                "t_stat_nw": s.t_stat_nw,
                "n_trials": n_trials,
                "t_hurdle_min": hurdle,
                "noise_floor": floor,
                "p_value": p,
                "fdr_pass": bool(fp),
                "passed": passed,
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------- #
# Stage 2: pristine OOS gate                                            #
# --------------------------------------------------------------------- #
def stage2_oos_gate(
    is_stats: ICStats, oos_stats: ICStats, cfg: GateConfig
) -> tuple[bool, dict]:
    """Pristine-OOS confirmation, evaluated exactly once per factor.

    Requires: retention |OOS|/|IS| >= cfg.min_retention with the same sign,
    OOS NW t >= cfg.min_oos_t, OOS win rate >= MIN_OOS_WIN_RATE, and the
    OOS |mean IC| clearing the per-horizon floor.  (McLean-Pontiff document
    ~32% IS->OOS decay on average; below 0.5 retention is overfit territory.)
    """
    from .splits import is_oos_retention

    details: dict[str, Any] = {
        "horizon_s": is_stats.horizon_s,
        "is_mean_ic": is_stats.mean_ic,
        "oos_mean_ic": oos_stats.mean_ic,
    }
    retention_ok = is_oos_retention(
        is_stats.mean_ic, oos_stats.mean_ic, min_retention=cfg.min_retention
    )
    retention = (
        abs(oos_stats.mean_ic) / abs(is_stats.mean_ic)
        if is_stats.mean_ic not in (0.0,) and math.isfinite(is_stats.mean_ic)
        else float("nan")
    )
    sign_ok = (
        math.isfinite(is_stats.mean_ic)
        and math.isfinite(oos_stats.mean_ic)
        and is_stats.mean_ic != 0.0
        and math.copysign(1.0, is_stats.mean_ic) == math.copysign(1.0, oos_stats.mean_ic)
    )
    t_ok = math.isfinite(oos_stats.t_stat_nw) and oos_stats.t_stat_nw >= cfg.min_oos_t
    win_ok = math.isfinite(oos_stats.win_rate) and oos_stats.win_rate >= MIN_OOS_WIN_RATE
    min_ic = cfg.min_rank_ic.get(is_stats.horizon_s, 0.02)
    level_ok = (
        math.isfinite(oos_stats.mean_ic) and abs(oos_stats.mean_ic) >= min_ic
    )

    details.update(
        {
            "retention": retention,
            "retention_ok": retention_ok,
            "sign_ok": sign_ok,
            "oos_t_ok": t_ok,
            "win_rate_ok": win_ok,
            "level_ok": level_ok,
        }
    )
    passed = bool(retention_ok and sign_ok and t_ok and win_ok and level_ok)
    return passed, details
