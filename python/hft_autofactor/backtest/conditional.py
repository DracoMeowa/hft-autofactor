"""Track-B conditional profitability matrix (#86).

Design spec: ``docs/design/eval-redesign-86.md`` (the "constitution" -- the
cell set, the primary-cell identity and the gate below are frozen there; do
not change them here without amending the spec and re-ledgering).

What this module answers
------------------------
Track A (``eval/gating.py``) decides whether a signal EXISTS and in which
direction.  Track B decides whether it can be TRADED for money: for each
(direction, entry-threshold tau, volatility-regime) cell it runs
non-overlapping, carry-free conditional trades on the 3s panel and reports
the net-of-cost edge in basis points.

Cell set (frozen, 24 per factor-horizon)
----------------------------------------
* direction in {long, short};
* tau in {0.005, 0.01, 0.05, 0.10}: enter only when |z| reaches the
  intraday trailing (1 - tau) quantile of |z|;
* regime in {all, vol_q80, vol_q90}: day must sit in the top 20% / 10% of
  TRAILING daily realized volatility (quantiles of the prior
  ``regime_window_days`` days -- never full-sample).

Given the track-A ``ic_direction``, the long cell trades the tail where the
factor predicts positive returns (tail sign = ic_direction) and the short
cell the opposite tail (tail sign = -ic_direction).  The PRIMARY cell is
(direction = long if ic_direction == +1 else short, tau = primary_tau,
regime = all); only it gates admission.  The other 23 cells are descriptive
evidence against post-hoc cell shopping.

Trade semantics (carry-free by construction)
--------------------------------------------
Entry at the actuation row (decision row = actuation - signal_lag_rows, the
same 1-row tradability margin as the engine), settlement at the first
snapshot with ts >= t + H -- the exact resolution point of the panel's
``fwd_mid_ret_{H}s`` label, so gross edge = label.  Greedy selection keeps
entries at least H apart, hence no overlapping windows, no position chains,
no structural carry: this is the deliberate contrast to the base-position
(底仓) full simulation whose iter-001 profits turned out to be pure carry
beta.

Costs per trade: taker spread crossing (+slippage ticks) with the
depth-impact overlay on both legs (same execution model as the engine), the
fee stack per commission scenario on both legs, and -- for shorts -- the
securities-lending borrow cost (:class:`ShortCostModel`).  While the borrow
parameters are not in the library (#129), short cells run uncosted but are
forced ``descriptive_only`` and can never pass the gate.

Causality discipline
--------------------
Every decision input is trailing: z from ``causal_zscore`` (per
instrument-day, no cross-day state), entry thresholds from an intraday
expanding quantile of |z| up to the decision row, regime quantiles from the
prior ``regime_window_days`` days.  A unit test perturbs future rows and
asserts that no entry decision or edge changes.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from ..eval.gating import TrialLedger, bhy_pass, p_value_two_sided
from ..eval.ic import newey_west_n_eff
from .costs import (
    HANDLING_FEE_EXEMPT_CATEGORIES,
    CostModel,
    ShortCostModel,
    short_borrow_cost_bps,
)
from .engine import _BAD_FLAG_MASK, _session_end_ts, _session_mask
from .execution import _ceil_tick, _floor_tick, cross_spread_fill, depth_impact_bps
from .signals import causal_zscore

__all__ = [
    "MatrixConfig",
    "MatrixResult",
    "cell_keys",
    "run_conditional_matrix",
    "write_matrix_report",
]

#: Regime names recognized by the frozen cell set and their vol quantiles.
_REGIME_QUANTILES = {"vol_q80": 80.0, "vol_q90": 90.0}
_ALL_REGIMES = ("all", "vol_q80", "vol_q90")


@dataclass(frozen=True)
class MatrixConfig:
    """Frozen cell-set + evaluation parameters of the conditional matrix.

    Defaults are the #86 pre-registration.  ``eval_dates`` semantics live on
    :func:`run_conditional_matrix`; note that the panel should contain
    ``regime_window_days`` days BEFORE the evaluation window so regime
    quantiles have trailing history.
    """

    taus: tuple[float, ...] = (0.005, 0.01, 0.05, 0.10)
    regimes: tuple[str, ...] = _ALL_REGIMES
    primary_tau: float = 0.01
    primary_regime: str = "all"
    z_window_rows: int = 100
    signal_lag_rows: int = 1  # tradability margin, engine parity (>= 1)
    intraday_warmup_rows: int = 1600  # ~80 min: no entries before this row
    regime_window_days: int = 20  # trailing days for vol-regime quantiles
    min_trades: int = 100  # validity floor per cell
    min_days: int = 10  # validity floor per cell (regime cells gated here)
    trade_units: int = 100_000  # fund units per conditional trade
    slippage_ticks: float = 1.0
    use_depth_impact: bool = True
    daily_t_max_lag: int = 5  # NW lag for the daily-mean edge t-stat
    min_primary_t: float = 2.0  # primary-cell gate: |t| floor on daily means
    fdr_q: float = 0.10  # BHY across cells (report column only)


@dataclass
class MatrixResult:
    """Full outcome of one (factor, horizon) conditional-matrix run."""

    factor: str
    horizon_s: int
    ic_direction: int
    cells: pl.DataFrame  # one row per (direction, tau, regime, scenario)
    primary_cell: dict  # {"direction", "tau", "regime"}
    primary: dict  # per-scenario verdicts + overall pass
    config_snapshot: dict
    regime_info: dict  # trailing-quantile warm-up diagnostics
    trades: pl.DataFrame  # trade-level audit dump (one row per trade/scenario)


def cell_keys(cfg: MatrixConfig) -> list[tuple[str, float, str]]:
    """The frozen cell set: every (direction, tau, regime) combination."""
    return [
        (d, tau, r)
        for d in ("long", "short")
        for tau in cfg.taus
        for r in cfg.regimes
    ]


def _validate(cfg: MatrixConfig) -> None:
    for tau in cfg.taus:
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0,1), got {tau}")
    if len(set(cfg.taus)) != len(cfg.taus):
        raise ValueError("taus must be distinct")
    for r in cfg.regimes:
        if r not in _ALL_REGIMES:
            raise ValueError(f"unknown regime {r!r}; expected subset of {_ALL_REGIMES}")
    if cfg.primary_tau not in cfg.taus:
        raise ValueError("primary_tau must be one of taus")
    if cfg.primary_regime not in cfg.regimes:
        raise ValueError("primary_regime must be one of regimes")
    if cfg.signal_lag_rows < 1:
        raise ValueError(
            f"signal_lag_rows must be >= 1 (tradability margin), "
            f"got {cfg.signal_lag_rows}"
        )
    if cfg.z_window_rows < 2:
        raise ValueError(f"z_window_rows must be >= 2, got {cfg.z_window_rows}")
    if cfg.intraday_warmup_rows < 1:
        raise ValueError("intraday_warmup_rows must be >= 1")
    if cfg.regime_window_days < 2:
        raise ValueError("regime_window_days must be >= 2")
    if cfg.trade_units <= 0:
        raise ValueError(f"trade_units must be > 0, got {cfg.trade_units}")


# --------------------------------------------------------------------- #
# per-day candidate trades                                              #
# --------------------------------------------------------------------- #
def _expanding_abs_z_thresholds(
    z: np.ndarray, taus: Sequence[float]
) -> dict[float, np.ndarray]:
    """Causal intraday (1 - tau) quantiles of |z| up to each decision row.

    Row ``j`` gets the quantile over |z[0..j]| (finite samples only) -- an
    expanding window, so no future row can influence any threshold.
    Implemented with a bisect-maintained sorted list: deterministic and
    O(n^2) only in cheap C memmoves.
    """
    n = z.shape[0]
    thr = {tau: np.full(n, np.nan) for tau in taus}
    sorted_abs: list[float] = []
    insort = bisect.insort
    for j in range(n):
        zj = float(z[j])
        if math.isfinite(zj):
            insort(sorted_abs, abs(zj))
        m = len(sorted_abs)
        if m >= 2:
            for tau in taus:
                idx = int(math.floor((1.0 - tau) * (m - 1)))
                thr[tau][j] = sorted_abs[idx]
    return thr


def _day_candidates(
    ts: np.ndarray,
    z: np.ndarray,
    tradable: np.ndarray,
    labels: np.ndarray,
    cfg: MatrixConfig,
    horizon_ms: int,
) -> dict[float, list[tuple[int, int]]]:
    """Non-overlapping greedy entry candidates per tau for one day.

    Returns ``tau -> [(actuation_row, tail_sign)]``.  A candidate at
    actuation row ``i`` needs: decision row ``j = i - lag`` at/past the
    intraday warm-up; ``z[j]`` finite, non-zero and |z[j]| above the causal
    (1 - tau) quantile computed on rows <= j; row i tradable with a usable
    label; and ts at least ``horizon_ms`` after the previously accepted
    entry (settlement of the previous trade).
    """
    n = ts.shape[0]
    lag = int(cfg.signal_lag_rows)
    thr = _expanding_abs_z_thresholds(z, cfg.taus)
    out: dict[float, list[tuple[int, int]]] = {tau: [] for tau in cfg.taus}
    for tau in cfg.taus:
        t_tau = thr[tau]
        last_ts = -(1 << 62)
        for i in range(lag, n):
            j = i - lag
            if j < cfg.intraday_warmup_rows:
                continue
            if not bool(tradable[i]):
                continue
            zj = float(z[j])
            if not math.isfinite(zj) or zj == 0.0:
                continue
            qj = float(t_tau[j])
            if not math.isfinite(qj) or abs(zj) < qj:
                continue
            if not math.isfinite(float(labels[i])):
                continue
            if int(ts[i]) - last_ts < horizon_ms:
                continue
            out[tau].append((i, 1 if zj > 0.0 else -1))
            last_ts = int(ts[i])
    return out


def _fills_for_row(
    side: str,
    bid: float,
    ask: float,
    bid_qty: float,
    ask_qty: float,
    units: int,
    cfg: MatrixConfig,
) -> float:
    """Taker fill at one row (engine semantics), NaN when not crossable."""
    fill = cross_spread_fill(side, bid, ask, slippage_ticks=cfg.slippage_ticks)
    if not math.isfinite(fill):
        return float("nan")
    if cfg.use_depth_impact:
        if side == "buy":
            best_cny = float(ask) * float(ask_qty)
        else:
            best_cny = float(bid) * float(bid_qty)
        bp = depth_impact_bps(float(units) * fill, best_cny)
        fill = fill * (1.0 + bp / 1e4) if side == "buy" else fill * (1.0 - bp / 1e4)
    return _ceil_tick(fill) if side == "buy" else _floor_tick(fill)


# --------------------------------------------------------------------- #
# fees (vectorized per scenario)                                        #
# --------------------------------------------------------------------- #
def _fee_bps_array(
    model: CostModel, fills: np.ndarray, units: int, exempt: np.ndarray
) -> np.ndarray:
    """Per-side fee-stack cost in bps of notional for an array of fills."""
    notional = np.asarray(fills, dtype=np.float64) * float(units)
    with np.errstate(divide="ignore", invalid="ignore"):
        comm = model.commission_rate * notional
        if model.min_commission_cny > 0.0:
            comm = np.maximum(comm, model.min_commission_cny)
        handling = np.where(
            exempt, 0.0, model.handling_fee_rate * notional
        )
        total = (
            comm
            + handling
            + model.regulatory_fee_rate * notional
            + model.transfer_fee_rate * notional
            + model.stamp_duty_rate * notional
        )
        bps = np.where(notional > 0.0, total / notional * 1e4, 0.0)
    return bps


# --------------------------------------------------------------------- #
# daily realized vol + trailing regime quantiles                        #
# --------------------------------------------------------------------- #
def _day_realized_vol(mid_session: np.ndarray) -> float:
    """Std of consecutive 3s mid returns within one session-filtered day."""
    m = mid_session[np.isfinite(mid_session) & (mid_session > 0)]
    if m.size < 30:
        return float("nan")
    r = m[1:] / m[:-1] - 1.0
    r = r[np.isfinite(r)]
    if r.size < 30:
        return float("nan")
    return float(np.std(r, ddof=1))


def _regime_thresholds_by_date(
    day_rv: Mapping[str, float], window_days: int
) -> dict[str, tuple[float, float] | None]:
    """Trailing (prior-window) q80/q90 thresholds of daily realized vol.

    Only days with a FULL ``window_days`` of prior history get thresholds;
    earlier days map to None and their trades can only join the ``all``
    regime cell.  Quantiles never see the current or future days.
    """
    dates = sorted(day_rv)
    out: dict[str, tuple[float, float] | None] = {}
    for pos, d in enumerate(dates):
        prior = [
            day_rv[x]
            for x in dates[max(0, pos - window_days): pos]
            if math.isfinite(day_rv[x])
        ]
        if len(prior) < window_days:
            out[d] = None
            continue
        arr = np.asarray(prior, dtype=np.float64)
        out[d] = (
            float(np.percentile(arr, _REGIME_QUANTILES["vol_q80"])),
            float(np.percentile(arr, _REGIME_QUANTILES["vol_q90"])),
        )
    return out


# --------------------------------------------------------------------- #
# the matrix runner                                                     #
# --------------------------------------------------------------------- #
def run_conditional_matrix(
    panel: pl.DataFrame,
    factor: str,
    horizon_s: int,
    cost_models: Mapping[str, CostModel],
    cfg: MatrixConfig,
    *,
    ic_direction: int,
    eval_dates: Sequence[str] | None = None,
    etf_categories: Mapping[str, str] | None = None,
    short_costs: ShortCostModel | None = None,
    ledger: TrialLedger | None = None,
) -> MatrixResult:
    """Run the frozen 24-cell conditional matrix of one (factor, horizon).

    ``panel`` must carry the interchange columns (date, exchange,
    instrument, ts_ms, flags, mid_px, bid1/ask1 px+qty), the factor column
    and ``fwd_mid_ret_{horizon_s}s``.  It should ALSO contain the
    ``regime_window_days`` trading days before ``eval_dates`` so the
    trailing volatility quantiles have history (rows outside
    ``eval_dates`` contribute to regime conditioning only, never to
    trades).  ``ic_direction`` is the track-A sign(mean IC) in {-1, +1}.

    When ``ledger`` is given, every cell is appended at stage="matrix_cell"
    BEFORE the primary gate is read (honest-N discipline, #86 section 6).
    """
    _validate(cfg)
    if ic_direction not in (-1, 1):
        raise ValueError(f"ic_direction must be +1 or -1, got {ic_direction}")
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be > 0, got {horizon_s}")
    if not cost_models:
        raise ValueError("cost_models must not be empty")

    col_l = f"fwd_mid_ret_{int(horizon_s)}s"
    required = [
        "date", "exchange", "instrument", "ts_ms", "flags", "mid_px",
        "last_px", "bid1_px", "ask1_px", "bid1_qty", "ask1_qty",
        factor, col_l,
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"panel lacks required columns: {missing}")

    eval_set = {str(d) for d in eval_dates} if eval_dates is not None else None
    df = panel.select(required).sort(["instrument", "date", "ts_ms"])
    horizon_ms = int(horizon_s) * 1000
    units = int(cfg.trade_units)
    categories = dict(etf_categories or {})

    trades: list[dict[str, Any]] = []
    regime_info: dict[str, Any] = {
        "window_days": int(cfg.regime_window_days),
        "instruments": {},
    }

    for (inst_raw,), idf in df.group_by("instrument", maintain_order=True):
        inst = str(inst_raw)
        exch = str(idf["exchange"][0])
        exempt = categories.get(inst, "equity_etf") in HANDLING_FEE_EXEMPT_CATEGORIES

        # -- pass 1: realized vol per day over ALL dates (regime history) --
        day_rv: dict[str, float] = {}
        day_frames: dict[str, pl.DataFrame] = {}
        for (date_raw,), ddf in idf.group_by("date", maintain_order=True):
            date = str(date_raw)
            ts_all = ddf["ts_ms"].cast(pl.Int64).to_numpy()
            smask = _session_mask(ts_all, exch)
            mid_s = ddf["mid_px"].cast(pl.Float64).to_numpy()[smask]
            day_rv[date] = _day_realized_vol(mid_s)
            day_frames[date] = ddf
        qthr = _regime_thresholds_by_date(day_rv, cfg.regime_window_days)
        n_with_regime = sum(1 for v in qthr.values() if v is not None)
        regime_info["instruments"][inst] = {
            "n_days_total": len(day_rv),
            "n_days_with_regime": n_with_regime,
            "rv_nan_days": sum(1 for v in day_rv.values() if not math.isfinite(v)),
        }

        # -- pass 2: candidate trades on evaluation days -------------------
        for date in sorted(day_frames):
            if eval_set is not None and date not in eval_set:
                continue
            ddf = day_frames[date]
            ts_all = ddf["ts_ms"].cast(pl.Int64).to_numpy()
            smask = _session_mask(ts_all, exch)
            if not bool(smask.any()):
                continue

            def col(name: str) -> np.ndarray:
                return ddf[name].cast(pl.Float64).to_numpy()[smask]

            ts = ts_all[smask].astype(np.int64)
            flags = ddf["flags"].cast(pl.Int64).to_numpy()[smask]
            bid1, ask1 = col("bid1_px"), col("ask1_px")
            mid = col("mid_px")
            last = col("last_px")
            bid_qty, ask_qty = col("bid1_qty"), col("ask1_qty")
            fvals = col(factor)
            labels = col(col_l)

            two_sided = (bid1 > 0) & (ask1 > 0) & (bid_qty > 0) & (ask_qty > 0)
            flags_ok = (flags & _BAD_FLAG_MASK) == 0
            end_ts = _session_end_ts(ts, exch)
            tradable = flags_ok & two_sided & ((end_ts - ts) >= horizon_ms)

            z = causal_zscore(fvals, int(cfg.z_window_rows))
            cands = _day_candidates(ts, z, tradable, labels, cfg, horizon_ms)

            qday = qthr.get(date)
            rv = day_rv.get(date, float("nan"))
            in_q80 = bool(qday is not None and math.isfinite(rv) and rv >= qday[0])
            in_q90 = bool(qday is not None and math.isfinite(rv) and rv >= qday[1])

            for tau in cfg.taus:
                for i, tail in cands[tau]:
                    m_in = float(mid[i])
                    if not (math.isfinite(m_in) and m_in > 0):
                        continue
                    k = int(np.searchsorted(ts, int(ts[i]) + horizon_ms, side="left"))
                    if k >= ts.shape[0]:
                        continue  # label resolution outside the day: skip
                    m_out = float(mid[k])
                    if not (math.isfinite(m_out) and m_out > 0):
                        continue
                    lin = _fills_for_row(
                        "buy", bid1[i], ask1[i], bid_qty[i], ask_qty[i], units, cfg
                    )
                    sin = _fills_for_row(
                        "sell", bid1[i], ask1[i], bid_qty[i], ask_qty[i], units, cfg
                    )
                    if not (math.isfinite(lin) and math.isfinite(sin)):
                        continue  # entry not executable despite two-sided flag
                    # Short-sale uptick rule (融资融券实施细则): a securities-
                    # lending sell order must be priced >= the last trade.
                    # When the crossed-bid fill undercuts it, the short entry
                    # degrades to posting at the (tick-rounded) last price.
                    last_i = float(last[i])
                    if math.isfinite(last_i) and last_i > 0:
                        sin = max(sin, _ceil_tick(last_i))
                    # Exit fills: taker at the resolution row; a one-sided
                    # resolution row degrades to a mid exit (zero exit slip).
                    lout = _fills_for_row(
                        "buy", bid1[k], ask1[k], bid_qty[k], ask_qty[k], units, cfg
                    )
                    sout = _fills_for_row(
                        "sell", bid1[k], ask1[k], bid_qty[k], ask_qty[k], units, cfg
                    )
                    if not math.isfinite(lout):
                        lout = m_out
                    if not math.isfinite(sout):
                        sout = m_out
                    gross = float(labels[i])
                    trades.append(
                        {
                            "date": date,
                            "instrument": inst,
                            "tau": float(tau),
                            "tail": int(tail),
                            "ts_in_ms": int(ts[i]),
                            "ts_out_ms": int(ts[k]),
                            "gross_ret": gross,
                            "mid_in": m_in,
                            "mid_out": m_out,
                            "long_fill_in": lin,
                            "long_fill_out": sout,
                            "short_fill_in": sin,
                            "short_fill_out": lout,
                            "in_regime_q80": in_q80,
                            "in_regime_q90": in_q90,
                            "exempt_category": exempt,
                        }
                    )

    tdf = pl.DataFrame(trades) if trades else pl.DataFrame(
        schema={
            "date": pl.String, "instrument": pl.String, "tau": pl.Float64,
            "tail": pl.Int8, "ts_in_ms": pl.Int64, "ts_out_ms": pl.Int64,
            "gross_ret": pl.Float64, "mid_in": pl.Float64, "mid_out": pl.Float64,
            "long_fill_in": pl.Float64, "long_fill_out": pl.Float64,
            "short_fill_in": pl.Float64, "short_fill_out": pl.Float64,
            "in_regime_q80": pl.Boolean, "in_regime_q90": pl.Boolean,
            "exempt_category": pl.Boolean,
        }
    )

    cells, audit = _aggregate_cells(
        tdf, cfg, ic_direction, cost_models, short_costs, int(horizon_s)
    )

    # -- honest-N: ledger every cell BEFORE the gate reads any threshold ---
    if ledger is not None:
        for row in cells.iter_rows(named=True):
            if row["scenario"] != next(iter(cost_models)):
                continue  # one ledger entry per cell, first scenario metrics
            ledger.log(
                factor=factor,
                horizon_s=int(horizon_s),
                params={
                    "direction": row["direction"],
                    "tau": row["tau"],
                    "regime": row["regime"],
                    "ic_direction": int(ic_direction),
                    "n_cells": len(cell_keys(cfg)),
                },
                stage="matrix_cell",
                metrics={
                    "n_trades": row["n_trades"],
                    "n_days": row["n_days"],
                    "mean_net_edge_bps": row["mean_net_edge_bps"],
                    "t_nw_daily": row["t_nw_daily"],
                    "valid": row["valid"],
                },
            )
    if ledger is not None and cells.height == 0:
        # Honest-N discipline: a factor that produces no trades at all is
        # still an evaluated trial and must not vanish from the ledger.
        ledger.log(
            factor=factor,
            horizon_s=int(horizon_s),
            params={
                "ic_direction": int(ic_direction),
                "n_cells": len(cell_keys(cfg)),
            },
            stage="matrix_cell",
            metrics={"n_trades": 0, "note": "no trades in any cell"},
        )

    primary_direction = "long" if ic_direction == 1 else "short"
    primary_cell = {
        "direction": primary_direction,
        "tau": float(cfg.primary_tau),
        "regime": str(cfg.primary_regime),
    }
    primary = _primary_gate(cells, primary_cell, cfg, short_costs is not None)

    config_snapshot = {
        "taus": list(cfg.taus),
        "regimes": list(cfg.regimes),
        "primary_cell": primary_cell,
        "z_window_rows": cfg.z_window_rows,
        "signal_lag_rows": cfg.signal_lag_rows,
        "intraday_warmup_rows": cfg.intraday_warmup_rows,
        "regime_window_days": cfg.regime_window_days,
        "min_trades": cfg.min_trades,
        "min_days": cfg.min_days,
        "trade_units": cfg.trade_units,
        "slippage_ticks": cfg.slippage_ticks,
        "use_depth_impact": cfg.use_depth_impact,
        "daily_t_max_lag": cfg.daily_t_max_lag,
        "min_primary_t": cfg.min_primary_t,
        "fdr_q": cfg.fdr_q,
        "ic_direction": int(ic_direction),
        "short_cost_model": (
            None
            if short_costs is None
            else {
                "borrow_rate_annual": short_costs.borrow_rate_annual,
                "min_charge_days": short_costs.min_charge_days,
                "day_count_base": short_costs.day_count_base,
                "source": short_costs.source,
            }
        ),
        # Regulatory reality check (#129): securities-lending shorts can only
        # be repaid from the next trading day -- the measured H-second edge
        # of a short cell therefore excludes the overnight gap exposure of
        # the mandatory overnight hold.  Documented, not priced.
        "short_settlement": (
            "T+1_repay: 买券还券 earliest next trading day; min hold = "
            "overnight; borrow >= 1 calendar day charged"
        ),
    }
    return MatrixResult(
        factor=factor,
        horizon_s=int(horizon_s),
        ic_direction=int(ic_direction),
        cells=cells,
        primary_cell=primary_cell,
        primary=primary,
        config_snapshot=config_snapshot,
        regime_info=regime_info,
        trades=audit,
    )


# --------------------------------------------------------------------- #
# aggregation                                                           #
# --------------------------------------------------------------------- #
def _daily_nw_t(daily_means: np.ndarray, max_lag: int) -> float:
    """One-sample Newey-West t of daily mean edges (zero when degenerate)."""
    arr = daily_means[np.isfinite(daily_means)]
    n = arr.size
    if n < 2:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std <= 0.0:
        return math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    n_eff = newey_west_n_eff(arr, max_lag=max_lag)
    return mean / (std / math.sqrt(n_eff))


def _cell_stats(
    net: np.ndarray, gross: np.ndarray, cost: np.ndarray, dates: np.ndarray,
    cfg: MatrixConfig,
) -> dict[str, float]:
    n_trades = int(net.size)
    if n_trades == 0:
        return {
            "n_trades": 0, "n_days": 0, "mean_net_edge_bps": float("nan"),
            "t_nw_daily": 0.0, "hit_rate": float("nan"),
            "gross_edge_bps": float("nan"), "cost_bps": float("nan"),
        }
    order = np.argsort(dates, kind="mergesort")
    d_sorted = dates[order]
    net_sorted = net[order]
    boundaries = np.flatnonzero(d_sorted[1:] != d_sorted[:-1]) + 1
    daily_sum = np.add.reduceat(net_sorted, np.concatenate(([0], boundaries)))
    daily_cnt = np.add.reduceat(np.ones_like(net_sorted), np.concatenate(([0], boundaries)))
    daily_means = daily_sum / daily_cnt
    return {
        "n_trades": n_trades,
        "n_days": int(daily_means.size),
        "mean_net_edge_bps": float(net.mean()),
        "t_nw_daily": float(_daily_nw_t(daily_means, cfg.daily_t_max_lag)),
        "hit_rate": float((net > 0).mean()),
        "gross_edge_bps": float(gross.mean()),
        "cost_bps": float(cost.mean()),
    }


def _aggregate_cells(
    tdf: pl.DataFrame,
    cfg: MatrixConfig,
    ic_direction: int,
    cost_models: Mapping[str, CostModel],
    short_costs: ShortCostModel | None,
    horizon_s: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Turn the trade dump into the long-form cell table + audit frame."""
    scen_names = list(cost_models)
    if tdf.height == 0:
        empty_cells = pl.DataFrame(
            schema={
                "direction": pl.String, "tau": pl.Float64, "regime": pl.String,
                "scenario": pl.String, "n_trades": pl.Int64, "n_days": pl.Int64,
                "mean_net_edge_bps": pl.Float64, "t_nw_daily": pl.Float64,
                "hit_rate": pl.Float64, "gross_edge_bps": pl.Float64,
                "cost_bps": pl.Float64, "valid": pl.Boolean,
                "descriptive_only": pl.Boolean, "bhy_pass": pl.Boolean,
                "cost_model": pl.String,
            }
        )
        return empty_cells, tdf

    n = tdf.height
    tau_arr = tdf["tau"].to_numpy()
    tail_arr = tdf["tail"].to_numpy().astype(np.int64)
    dates_arr = np.asarray(tdf["date"].to_list())
    gross_ret = tdf["gross_ret"].to_numpy().astype(np.float64)
    mid_in = tdf["mid_in"].to_numpy().astype(np.float64)
    mid_out = tdf["mid_out"].to_numpy().astype(np.float64)
    exempt = tdf["exempt_category"].to_numpy().astype(bool)
    in_q80 = tdf["in_regime_q80"].to_numpy().astype(bool)
    in_q90 = tdf["in_regime_q90"].to_numpy().astype(bool)
    units = int(cfg.trade_units)

    # direction of each trade given ic_direction (tail -> cell, spec 4.1)
    dir_arr = np.where(tail_arr == ic_direction, "long", "short")

    fills = {
        ("long", "in"): tdf["long_fill_in"].to_numpy().astype(np.float64),
        ("long", "out"): tdf["long_fill_out"].to_numpy().astype(np.float64),
        ("short", "in"): tdf["short_fill_in"].to_numpy().astype(np.float64),
        ("short", "out"): tdf["short_fill_out"].to_numpy().astype(np.float64),
    }
    mid_ref = {("in"): mid_in, ("out"): mid_out}
    slip = {
        key: np.abs(fills[key] - mid_ref[leg]) / mid_ref[leg] * 1e4
        for key, leg in ((("long", "in"), "in"), (("long", "out"), "out"),
                         (("short", "in"), "in"), (("short", "out"), "out"))
    }
    gross_bps = {
        "long": gross_ret * 1e4,
        "short": -gross_ret * 1e4,
    }
    borrow_bps_short = (
        short_borrow_cost_bps(short_costs, float(horizon_s))
        if short_costs is not None
        else None
    )

    audit_cols = {c: tdf[c] for c in tdf.columns}
    rows: list[dict[str, Any]] = []

    for scen in scen_names:
        model = cost_models[scen]
        fee = {
            key: _fee_bps_array(model, fills[key], units, exempt)
            for key in (("long", "in"), ("long", "out"),
                        ("short", "in"), ("short", "out"))
        }
        net = {
            d: gross_bps[d] - slip[(d, "in")] - slip[(d, "out")]
            - fee[(d, "in")] - fee[(d, "out")]
            - (borrow_bps_short if d == "short" and borrow_bps_short is not None else 0.0)
            for d in ("long", "short")
        }
        cost_total = {
            d: slip[(d, "in")] + slip[(d, "out")] + fee[(d, "in")] + fee[(d, "out")]
            + (borrow_bps_short if d == "short" and borrow_bps_short is not None else 0.0)
            for d in ("long", "short")
        }
        if scen == scen_names[0]:
            for d in ("long", "short"):
                audit_cols[f"net_edge_bps_{d}"] = net[d]
                audit_cols[f"cost_bps_{d}"] = cost_total[d]

        for direction, tau, regime in cell_keys(cfg):
            mask = (dir_arr == direction) & (tau_arr == tau)
            if regime == "vol_q80":
                mask &= in_q80
            elif regime == "vol_q90":
                mask &= in_q90
            sel_net = net[direction][mask]
            stats = _cell_stats(
                sel_net, gross_bps[direction][mask], cost_total[direction][mask],
                dates_arr[mask], cfg,
            )
            valid = bool(
                stats["n_trades"] >= cfg.min_trades
                and stats["n_days"] >= cfg.min_days
                and math.isfinite(stats["mean_net_edge_bps"])
            )
            short_uncosted = direction == "short" and short_costs is None
            if short_uncosted:
                valid = False
            cost_model = (
                "taker_v1"
                if direction == "long"
                else ("taker_v1+borrow" if short_costs is not None else "taker_v1_no_borrow_PARAMS_PENDING")
            )
            rows.append(
                {
                    "direction": direction,
                    "tau": float(tau),
                    "regime": regime,
                    "scenario": scen,
                    "n_trades": int(stats["n_trades"]),
                    "n_days": int(stats["n_days"]),
                    "mean_net_edge_bps": stats["mean_net_edge_bps"],
                    "t_nw_daily": stats["t_nw_daily"],
                    "hit_rate": stats["hit_rate"],
                    "gross_edge_bps": stats["gross_edge_bps"],
                    "cost_bps": stats["cost_bps"],
                    "valid": valid,
                    "descriptive_only": not valid,
                    "bhy_pass": False,  # filled below
                    "cost_model": cost_model,
                }
            )

    cells = pl.DataFrame(rows)

    # BHY across the 24 cells on the first scenario's p-values (report
    # column only; the primary gate never reads it -- spec section 6.3).
    first = scen_names[0]
    cell_ids = cell_keys(cfg)
    p_list: list[float] = []
    for row in cells.iter_rows(named=True):
        if row["scenario"] != first:
            continue
        p_list.append(p_value_two_sided(row["t_nw_daily"]))
    if p_list:
        passed = bhy_pass(p_list, cfg.fdr_q)
        bhy_by_cell = {cid: bool(b) for cid, b in zip(cell_ids, passed)}
        cells = cells.with_columns(
            pl.Series(
                "bhy_pass",
                [bhy_by_cell[(r["direction"], r["tau"], r["regime"])] for r in cells.iter_rows(named=True)],
            )
        )

    audit = pl.DataFrame(audit_cols)
    return cells, audit


def _primary_gate(
    cells: pl.DataFrame,
    primary_cell: dict,
    cfg: MatrixConfig,
    short_model_available: bool,
) -> dict:
    """Spec 4.7: the primary cell must clear the net-cost gate in ALL
    commission scenarios (plus the borrow-model availability rule)."""
    verdicts: dict[str, Any] = {}
    all_ok = True
    if cells.height == 0:
        return {"passed": False, "reason": "no trades in any cell", "scenarios": {}}
    for row in cells.iter_rows(named=True):
        if (
            row["direction"] != primary_cell["direction"]
            or row["tau"] != primary_cell["tau"]
            or row["regime"] != primary_cell["regime"]
        ):
            continue
        scen = row["scenario"]
        reasons: list[str] = []
        mean_net = row["mean_net_edge_bps"]
        t = row["t_nw_daily"]
        if not row["valid"]:
            reasons.append(
                f"cell not valid (n_trades={row['n_trades']}<min or "
                f"n_days={row['n_days']}<min"
                + ("; short borrow parameters pending #129"
                   if row["cost_model"].endswith("PARAMS_PENDING") else "")
                + ")"
            )
        if not (math.isfinite(mean_net) and mean_net > 0.0):
            reasons.append(f"mean net edge {mean_net:.3f} bp <= 0")
        if not (math.isfinite(t) and t >= cfg.min_primary_t):
            reasons.append(f"daily NW t {t:.2f} < {cfg.min_primary_t}")
        if math.isfinite(row["gross_edge_bps"]) and math.isfinite(row["cost_bps"]):
            if not row["gross_edge_bps"] > row["cost_bps"]:
                reasons.append(
                    f"gross edge {row['gross_edge_bps']:.3f} bp does not cover "
                    f"cost {row['cost_bps']:.3f} bp"
                )
        if row["direction"] == "short" and not short_model_available:
            reasons.append("short borrow cost model unavailable (#129 pending)")
        ok = not reasons
        all_ok = all_ok and ok
        verdicts[scen] = {
            "pass": ok,
            "reasons": reasons,
            "mean_net_edge_bps": mean_net,
            "t_nw_daily": t,
            "n_trades": row["n_trades"],
            "n_days": row["n_days"],
        }
    return {
        "passed": bool(all_ok and verdicts),
        "cell": primary_cell,
        "scenarios": verdicts,
    }


# --------------------------------------------------------------------- #
# report writing                                                        #
# --------------------------------------------------------------------- #
def write_matrix_report(out_dir: str | Path, result: MatrixResult) -> Path:
    """Write matrix.json / matrix.md / trades.csv under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "factor": result.factor,
        "horizon_s": result.horizon_s,
        "ic_direction": result.ic_direction,
        "config": result.config_snapshot,
        "primary_cell": result.primary_cell,
        "primary_gate": result.primary,
        "regime_info": result.regime_info,
        "cells": result.cells.to_dicts(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "matrix.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if result.trades.height:
        result.trades.write_csv(out / "trades.csv")

    lines: list[str] = []
    lines.append(
        f"# 条件盈利矩阵 — {result.factor} @ {result.horizon_s}s"
    )
    lines.append("")
    lines.append(
        f"- ic_direction: {result.ic_direction:+d} · 主格: "
        f"{result.primary_cell['direction']} / tau={result.primary_cell['tau']} / "
        f"{result.primary_cell['regime']}"
    )
    lines.append(
        f"- 主格 gate: **{'PASS' if result.primary.get('passed') else 'FAIL'}**"
    )
    for scen, v in result.primary.get("scenarios", {}).items():
        lines.append(
            f"  - {scen}: {'PASS' if v['pass'] else 'FAIL'} · net="
            f"{v['mean_net_edge_bps']:.3f}bp · t={v['t_nw_daily']:.2f} · "
            f"n={v['n_trades']}/{v['n_days']}d"
        )
        for r in v.get("reasons", []):
            lines.append(f"    - {r}")
    lines.append("")
    lines.append("| direction | tau | regime | scenario | n | days | net(bp) | t | gross(bp) | cost(bp) | valid | bhy |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in result.cells.iter_rows(named=True):
        net = row["mean_net_edge_bps"]
        gross = row["gross_edge_bps"]
        cost = row["cost_bps"]
        lines.append(
            f"| {row['direction']} | {row['tau']} | {row['regime']} | "
            f"{row['scenario']} | {row['n_trades']} | {row['n_days']} | "
            f"{net:.3f} | {row['t_nw_daily']:.2f} | "
            f"{gross if math.isfinite(gross) else float('nan'):.3f} | "
            f"{cost if math.isfinite(cost) else float('nan'):.3f} | "
            f"{row['valid']} | {row['bhy_pass']} |"
        )
    (out / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "matrix.json"
