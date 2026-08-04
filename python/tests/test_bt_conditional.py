"""Tests for the track-B conditional profitability matrix (#86).

Covers:
* expanding |z| entry thresholds are strictly causal (future-proof);
* future-row perturbation leaves earlier entry decisions unchanged;
* a factor with a known long edge passes the primary cell in ALL commission
  scenarios while its short leg loses money (spec section 8.1);
* a reversal factor (negative IC, direction=-1) passes with a SHORT primary
  cell once borrow costs are modelled;
* a pure-noise factor fails the gate with negative net edge;
* non-overlapping trade spacing (>= horizon between entries);
* regime cells are descriptive-only until enough trailing history exists;
* securities-lending borrow math (min 1 calendar day, 360 base);
* short-sale uptick rule (entry fill >= last price);
* short cells without a borrow model are forced descriptive-only;
* the trial ledger receives one matrix_cell entry per cell BEFORE the gate.

Synthetic panels are generated with known per-row forward labels so the
expected edge sign of each cell is analytic.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hft_autofactor.backtest.conditional import (
    MatrixConfig,
    _expanding_abs_z_thresholds,
    cell_keys,
    run_conditional_matrix,
    write_matrix_report,
)
from hft_autofactor.backtest.costs import (
    CostModel,
    ShortCostModel,
    short_borrow_cost_bps,
)
from hft_autofactor.eval.gating import TrialLedger

HORIZON_S = 60
ROW_MS = 3000
SESSION_START_MS = 34_200_000


# --------------------------------------------------------------------------- #
# cost models                                                                 #
# --------------------------------------------------------------------------- #
def _cost_models() -> dict[str, CostModel]:
    return {
        "institutional": CostModel(
            name="institutional", commission_rate=0.00005, min_commission_cny=0.0
        ),
        "retail_negotiated": CostModel(
            name="retail_negotiated", commission_rate=0.0001, min_commission_cny=5.0
        ),
        "retail_default": CostModel(
            name="retail_default", commission_rate=0.00025, min_commission_cny=5.0
        ),
    }


def _cfg(**overrides) -> MatrixConfig:
    """Small-panel test config: same frozen cell set, relaxed validity floors."""
    base = dict(
        intraday_warmup_rows=120,
        min_trades=15,
        min_days=5,
        regime_window_days=5,
        z_window_rows=100,
    )
    base.update(overrides)
    return MatrixConfig(**base)


# --------------------------------------------------------------------------- #
# synthetic panel                                                             #
# --------------------------------------------------------------------------- #
def _synthetic_panel(
    *,
    n_days: int = 15,
    rows_per_day: int = 800,
    edge_bps: float = 30.0,
    label_mode: str = "up",  # "up" | "down" | "symmetric"
    factor_sign: float = 1.0,  # +1 predictor, -1 reversal
    noise_factor: bool = False,
    vol_scales: list[float] | None = None,
    last_offset_ticks: int = 0,
    seed: int = 0,
    start: str = "2025-01-06",  # a Monday
) -> tuple[pl.DataFrame, list[str]]:
    """Panel with a controlled per-row forward label.

    ``s`` in {-1,+1} is the latent direction per row; the factor is
    ``factor_sign * s * g`` with a random magnitude ``g`` (or pure noise),
    so the sign of its z-score reveals ``s`` while the entry THRESHOLD (the
    expanding |z| quantile) selects on magnitude.

    The label of row ``r`` is driven by ``s[r-1]``, NOT ``s[r]``: the matrix
    decides at row ``j`` and actuates one row later (``signal_lag_rows=1``),
    so a factor that is predictive under execution lag carries its edge on
    the ACTUATION row's forward return.  Labeling by the same row would let
    the lag destroy the edge and mis-test the module.
    """
    rng = np.random.default_rng(seed)
    d0 = dt.date.fromisoformat(start)
    dates: list[str] = []
    offset = 0
    while len(dates) < n_days:
        d = d0 + dt.timedelta(days=offset)
        offset += 1
        if d.weekday() < 5:
            dates.append(d.strftime("%Y%m%d"))

    records: list[dict] = []
    for di, date in enumerate(dates):
        vol_scale = 1.0 if vol_scales is None else vol_scales[di]
        s = rng.choice([-1.0, 1.0], size=rows_per_day)
        g = rng.exponential(1.0, size=rows_per_day) + 0.3
        if noise_factor:
            f = rng.standard_normal(rows_per_day)
        else:
            f = factor_sign * s * g
        mid = 4.0 + np.cumsum(rng.standard_normal(rows_per_day)) * 2e-4 * vol_scale
        mid = np.clip(mid, 3.5, 4.5)
        # edge lives on the row AFTER the signal (actuation row, lag = 1)
        s_prev = np.concatenate((np.array([0.0]), s[:-1]))
        if label_mode == "up":
            label = np.where(s_prev > 0, edge_bps, 0.0)
        elif label_mode == "down":
            label = np.where(s_prev < 0, -edge_bps, 0.0)
        elif label_mode == "symmetric":
            label = s_prev * edge_bps
        else:
            raise ValueError(label_mode)
        label = label * 1e-4 * (1.0 + 0.1 * rng.standard_normal(rows_per_day))
        for r in range(rows_per_day):
            m = float(mid[r])
            records.append(
                {
                    "date": date,
                    "exchange": "sse",
                    "instrument": "588000",
                    "ts_ms": SESSION_START_MS + r * ROW_MS,
                    "flags": 0,
                    "mid_px": m,
                    "last_px": m + last_offset_ticks * 0.001,
                    "bid1_px": m - 0.001,
                    "ask1_px": m + 0.001,
                    "bid1_qty": 1.0e7,
                    "ask1_qty": 1.0e7,
                    "factor": float(f[r]),
                    f"fwd_mid_ret_{HORIZON_S}s": float(label[r]),
                }
            )
    return pl.DataFrame(records), dates


def _cell(result, direction: str, tau: float, regime: str, scenario: str):
    sel = result.cells.filter(
        (pl.col("direction") == direction)
        & (pl.col("tau") == tau)
        & (pl.col("regime") == regime)
        & (pl.col("scenario") == scenario)
    )
    assert sel.height == 1
    return sel.row(0, named=True)


# --------------------------------------------------------------------------- #
# causal entry thresholds                                                     #
# --------------------------------------------------------------------------- #
def test_expanding_thresholds_are_future_proof():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(300)
    thr = _expanding_abs_z_thresholds(z, (0.01, 0.10))
    z2 = z.copy()
    z2[200:] = 50.0  # extreme future values
    thr2 = _expanding_abs_z_thresholds(z2, (0.01, 0.10))
    for tau in (0.01, 0.10):
        assert np.allclose(thr[tau][:200], thr2[tau][:200], equal_nan=True)
        # and the contaminated tail does move its own thresholds
        assert not np.allclose(thr[tau][250:], thr2[tau][250:], equal_nan=True)


def test_expanding_threshold_known_value():
    z = np.array([0.0, -1.0, 2.0, -3.0])
    thr = _expanding_abs_z_thresholds(z, (0.25,))
    # row 3: sorted |z| = [0,1,2,3], idx = floor(0.75 * 3) = 2 -> 2.0
    assert thr[0.25][3] == pytest.approx(2.0)
    assert math.isnan(thr[0.25][0])  # one sample: no quantile


# --------------------------------------------------------------------------- #
# long-edge factor: primary long cell passes, short leg loses                 #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def long_edge_result():
    panel, dates = _synthetic_panel(label_mode="up", factor_sign=1.0, seed=11)
    return run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(),
        ic_direction=1,
        eval_dates=dates,
        short_costs=ShortCostModel(borrow_rate_annual=0.08),
    )


def test_long_primary_cell_passes_all_scenarios(long_edge_result):
    res = long_edge_result
    assert res.primary_cell == {"direction": "long", "tau": 0.01, "regime": "all"}
    assert res.primary["passed"] is True
    for scen in ("institutional", "retail_negotiated", "retail_default"):
        cell = _cell(res, "long", 0.01, "all", scen)
        assert cell["valid"] is True
        assert cell["mean_net_edge_bps"] > 10.0  # ~30bp edge - ~12-16bp cost
        assert cell["t_nw_daily"] >= 2.0
        assert cell["gross_edge_bps"] > cell["cost_bps"]
        assert res.primary["scenarios"][scen]["pass"] is True


def test_short_leg_of_long_factor_loses_money(long_edge_result):
    res = long_edge_result
    for scen in ("institutional", "retail_negotiated", "retail_default"):
        cell = _cell(res, "short", 0.01, "all", scen)
        assert cell["n_trades"] > 0
        # shorting the rows that go nowhere gross ~= 0, so net ~= -cost
        assert cell["mean_net_edge_bps"] < 0.0
        assert cell["gross_edge_bps"] < cell["cost_bps"]


def test_cell_set_is_frozen_24(long_edge_result):
    assert len(cell_keys(MatrixConfig())) == 24
    assert long_edge_result.cells.height == 24 * 3  # 24 cells x 3 scenarios


# --------------------------------------------------------------------------- #
# reversal factor: direction=-1, short primary passes once borrow is costed   #
# --------------------------------------------------------------------------- #
def test_reversal_factor_short_primary_passes():
    panel, dates = _synthetic_panel(label_mode="down", factor_sign=-1.0, seed=22)
    res = run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(),
        ic_direction=-1,
        eval_dates=dates,
        short_costs=ShortCostModel(borrow_rate_annual=0.08),
    )
    assert res.primary_cell == {"direction": "short", "tau": 0.01, "regime": "all"}
    assert res.primary["passed"] is True
    cell = _cell(res, "short", 0.01, "all", "institutional")
    assert cell["cost_model"] == "taker_v1+borrow"
    assert cell["mean_net_edge_bps"] > 5.0  # 30bp edge - costs - ~2.2bp borrow
    # the opposite (long) cell trades the zero-label rows
    long_cell = _cell(res, "long", 0.01, "all", "institutional")
    assert long_cell["mean_net_edge_bps"] < 0.0


def test_short_cell_without_borrow_model_never_passes():
    panel, dates = _synthetic_panel(label_mode="down", factor_sign=-1.0, seed=22)
    res = run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(),
        ic_direction=-1,
        eval_dates=dates,
        short_costs=None,
    )
    cell = _cell(res, "short", 0.01, "all", "institutional")
    assert cell["valid"] is False
    assert cell["descriptive_only"] is True
    assert cell["cost_model"] == "taker_v1_no_borrow_PARAMS_PENDING"
    assert res.primary["passed"] is False
    reasons = " ".join(res.primary["scenarios"]["institutional"]["reasons"])
    assert "borrow" in reasons


# --------------------------------------------------------------------------- #
# noise factor: net edge negative, gate fails                                 #
# --------------------------------------------------------------------------- #
def test_noise_factor_fails_primary_gate():
    panel, dates = _synthetic_panel(
        label_mode="symmetric", noise_factor=True, seed=33
    )
    res = run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(),
        ic_direction=1,
        eval_dates=dates,
        short_costs=ShortCostModel(borrow_rate_annual=0.08),
    )
    assert res.primary["passed"] is False
    cell = _cell(res, "long", 0.01, "all", "institutional")
    # gross edge of random entries is ~0, so net = -costs
    assert cell["mean_net_edge_bps"] < 0.0
    assert cell["gross_edge_bps"] < cell["cost_bps"]


# --------------------------------------------------------------------------- #
# trade semantics: non-overlap, settlement, uptick                            #
# --------------------------------------------------------------------------- #
def test_trades_never_overlap(long_edge_result):
    res = long_edge_result
    horizon_ms = HORIZON_S * 1000
    for (date, tau), sub in res.trades.group_by(["date", "tau"]):
        sub = sub.sort("ts_in_ms")
        ts_in = sub["ts_in_ms"].to_numpy()
        ts_out = sub["ts_out_ms"].to_numpy()
        assert (ts_out - ts_in >= horizon_ms).all()
        assert (np.diff(ts_in) >= horizon_ms).all()


def test_uptick_rule_lifts_short_entries_to_last_price():
    panel, dates = _synthetic_panel(
        label_mode="up", factor_sign=1.0, last_offset_ticks=5, seed=44
    )
    res = run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(),
        ic_direction=1,
        eval_dates=dates,
        short_costs=ShortCostModel(borrow_rate_annual=0.08),
    )
    assert res.trades.height > 0
    short_in = res.trades["short_fill_in"].to_numpy()
    long_in = res.trades["long_fill_in"].to_numpy()
    mid_in = res.trades["mid_in"].to_numpy()
    # last = mid + 5 ticks: the crossed-bid fill (mid - 2 ticks) is below the
    # last trade, so the uptick rule re-prices every short entry >= last.
    assert (short_in >= mid_in + 0.004).all()
    # long entries are unaffected (cross ask + slippage, ceil to tick: the
    # mid is off-grid, so the fill sits in [mid+2 ticks, mid+3 ticks))
    assert (long_in >= mid_in + 0.002 - 1e-9).all()
    assert (long_in <= mid_in + 0.003 + 1e-9).all()


# --------------------------------------------------------------------------- #
# regime conditioning: trailing quantiles + validity flags                    #
# --------------------------------------------------------------------------- #
def test_regime_cells_subset_of_all_and_history_gated():
    panel, dates = _synthetic_panel(
        n_days=16,
        label_mode="up",
        factor_sign=1.0,
        vol_scales=[1.0, 3.0] * 8,  # alternating low/high vol days
        seed=55,
    )
    res = run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(regime_window_days=5),
        ic_direction=1,
        eval_dates=dates,
    )
    info = res.regime_info["instruments"]["588000"]
    assert info["n_days_total"] == 16
    # days 0..4 lack a full trailing window -> only 11 regime-eligible days
    assert info["n_days_with_regime"] == 11
    all_cell = _cell(res, "long", 0.05, "all", "institutional")
    q80 = _cell(res, "long", 0.05, "vol_q80", "institutional")
    q90 = _cell(res, "long", 0.05, "vol_q90", "institutional")
    assert q80["n_trades"] <= all_cell["n_trades"]
    assert q90["n_trades"] <= q80["n_trades"]
    assert q80["n_trades"] > 0  # high-vol days do contribute trades


def test_regime_cells_empty_without_history():
    panel, dates = _synthetic_panel(n_days=4, label_mode="up", factor_sign=1.0,
                                    seed=66)
    res = run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(regime_window_days=20, min_days=2, min_trades=5),
        ic_direction=1,
        eval_dates=dates,
    )
    assert res.regime_info["instruments"]["588000"]["n_days_with_regime"] == 0
    for regime in ("vol_q80", "vol_q90"):
        cell = _cell(res, "long", 0.05, regime, "institutional")
        assert cell["n_trades"] == 0
        assert cell["valid"] is False
        assert cell["descriptive_only"] is True


def test_validity_floor_marks_cells_descriptive_only():
    panel, dates = _synthetic_panel(label_mode="up", factor_sign=1.0, seed=77)
    res = run_conditional_matrix(
        panel,
        "factor",
        HORIZON_S,
        _cost_models(),
        _cfg(min_trades=10_000),  # nobody can pass this floor
        ic_direction=1,
        eval_dates=dates,
    )
    assert (res.cells["valid"].to_numpy() == False).all()  # noqa: E712
    assert (res.cells["descriptive_only"].to_numpy() == True).all()  # noqa: E712
    assert res.primary["passed"] is False
    reasons = " ".join(res.primary["scenarios"]["institutional"]["reasons"])
    assert "not valid" in reasons


# --------------------------------------------------------------------------- #
# future-perturbation invariance of entry decisions                           #
# --------------------------------------------------------------------------- #
def test_future_factor_perturbation_changes_nothing_before_it():
    panel, dates = _synthetic_panel(label_mode="up", factor_sign=1.0, seed=88)
    res_a = run_conditional_matrix(
        panel, "factor", HORIZON_S, _cost_models(), _cfg(),
        ic_direction=1, eval_dates=dates,
    )
    # perturb the factor on the LAST 100 rows of the LAST day only
    perturb_start_ts = SESSION_START_MS + (800 - 100) * ROW_MS
    last_day = dates[-1]
    perturbed = panel.with_columns(
        pl.when(
            (pl.col("date") == last_day) & (pl.col("ts_ms") >= perturb_start_ts)
        )
        .then(pl.col("factor") * 10.0 + 7.0)
        .otherwise(pl.col("factor"))
        .alias("factor")
    )
    res_b = run_conditional_matrix(
        perturbed, "factor", HORIZON_S, _cost_models(), _cfg(),
        ic_direction=1, eval_dates=dates,
    )

    def prefix(res):
        t = res.trades
        return t.filter(
            (pl.col("date") != last_day) | (pl.col("ts_in_ms") < perturb_start_ts)
        ).select(["date", "tau", "tail", "ts_in_ms", "ts_out_ms", "gross_ret"])

    pa, pb = prefix(res_a), prefix(res_b)
    assert pa.height == pb.height
    assert pa.sort(["date", "tau", "ts_in_ms"]).equals(
        pb.sort(["date", "tau", "ts_in_ms"])
    )


# --------------------------------------------------------------------------- #
# ledger + reports                                                            #
# --------------------------------------------------------------------------- #
def test_ledger_records_every_cell_before_gate(tmp_path):
    panel, dates = _synthetic_panel(label_mode="up", factor_sign=1.0, seed=99)
    ledger = TrialLedger(tmp_path / "trial_ledger.jsonl")
    run_conditional_matrix(
        panel, "factor", HORIZON_S, _cost_models(), _cfg(),
        ic_direction=1, eval_dates=dates, ledger=ledger,
    )
    assert ledger.n_trials("matrix_cell") == 24  # one per cell, first scenario


def test_ledger_records_no_trade_runs(tmp_path):
    panel, dates = _synthetic_panel(n_days=2, rows_per_day=50, seed=101)
    ledger = TrialLedger(tmp_path / "trial_ledger.jsonl")
    res = run_conditional_matrix(
        panel, "factor", HORIZON_S, _cost_models(), _cfg(),
        ic_direction=1, eval_dates=dates, ledger=ledger,
    )
    assert res.trades.height == 0
    assert res.primary["passed"] is False
    assert ledger.n_trials("matrix_cell") == 1  # the no-trades summary entry


def test_write_matrix_report_artifacts(tmp_path, long_edge_result):
    out = write_matrix_report(tmp_path / "matrix", long_edge_result)
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["factor"] == "factor"
    assert payload["ic_direction"] == 1
    assert len(payload["cells"]) == 24 * 3
    assert payload["config"]["short_settlement"].startswith("T+1_repay")
    assert (tmp_path / "matrix" / "matrix.md").is_file()
    assert (tmp_path / "matrix" / "trades.csv").is_file()


# --------------------------------------------------------------------------- #
# input validation                                                            #
# --------------------------------------------------------------------------- #
def test_input_validation():
    panel, dates = _synthetic_panel(n_days=3, seed=111)
    models = _cost_models()
    with pytest.raises(ValueError):
        run_conditional_matrix(
            panel, "factor", HORIZON_S, models, _cfg(), ic_direction=0,
        )
    with pytest.raises(ValueError):
        run_conditional_matrix(
            panel.drop("bid1_px"), "factor", HORIZON_S, models, _cfg(),
            ic_direction=1,
        )
    with pytest.raises(ValueError):
        run_conditional_matrix(
            panel, "factor", HORIZON_S, {}, _cfg(), ic_direction=1,
        )
    with pytest.raises(ValueError):
        run_conditional_matrix(
            panel, "missing_factor", HORIZON_S, models, _cfg(), ic_direction=1,
        )
    with pytest.raises(ValueError):
        # primary_tau outside the frozen tau set is rejected at run time
        run_conditional_matrix(
            panel, "factor", HORIZON_S, models, MatrixConfig(primary_tau=0.5),
            ic_direction=1,
        )


# --------------------------------------------------------------------------- #
# borrow cost math (#129 parameters)                                          #
# --------------------------------------------------------------------------- #
def test_short_borrow_cost_min_one_day():
    m = ShortCostModel(borrow_rate_annual=0.08)
    # 900s hold is billed as ONE calendar day over a 360 base
    assert short_borrow_cost_bps(m, 900.0) == pytest.approx(0.08 / 360.0 * 1e4)
    assert short_borrow_cost_bps(m, 15.0) == pytest.approx(0.08 / 360.0 * 1e4)
    # two calendar days of borrow
    assert short_borrow_cost_bps(m, 2 * 86_400.0) == pytest.approx(
        2 * 0.08 / 360.0 * 1e4
    )


def test_short_borrow_cost_validation():
    m = ShortCostModel(borrow_rate_annual=0.08)
    with pytest.raises(ValueError):
        short_borrow_cost_bps(m, -1.0)
