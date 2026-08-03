"""Engine tests: tick-rounded fills, depth-impact overlay, price limits,
session-exclusion windows, PnL arithmetic, and the multi-scenario metrics.

All tests run on fully synthetic days so every number can be hand-checked.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_autofactor.backtest.costs import CostModel
from hft_autofactor.backtest.engine import (
    BacktestResult,
    DaySim,
    InstrumentMeta,
    run_backtest,
    simulate_day,
)
from hft_autofactor.backtest.execution import (
    TICK_CNY,
    clamp_price_limit,
    cross_spread_fill,
    depth_impact_bps,
)
from hft_autofactor.backtest.metrics import gate_on_costs, summarize_results
from hft_autofactor.backtest.signals import (
    PositionRule,
    causal_zscore,
    position_from_z,
    zscore_column,
)

ZERO_COSTS = CostModel(
    name="zero",
    commission_rate=0.0,
    min_commission_cny=0.0,
    handling_fee_rate=0.0,
    regulatory_fee_rate=0.0,
    transfer_fee_rate=0.0,
    stamp_duty_rate=0.0,
)

RULE = PositionRule(
    entry_z=2.0, exit_z=0.5, direction=1, max_position_units=1000, signal_lag_rows=1
)

T0_META = InstrumentMeta(exchange="sse", settlement="T+0", etf_category="bond_etf")
T1_META = InstrumentMeta(exchange="sse", settlement="T+1", etf_category="equity_etf")


def make_day(
    n: int = 6,
    *,
    z: list[float],
    bid: float = 4.000,
    ask: float = 4.001,
    mid: float | list[float] | None = None,
    ask_qty: float = 1e9,
    bid_qty: float = 1e9,
    tradable: list[bool] | None = None,
) -> DaySim:
    default_mid = 0.5 * (bid + ask)
    if mid is None:
        mid_arr = np.full(n, default_mid)
    elif isinstance(mid, list):
        mid_arr = np.asarray(mid, dtype=np.float64)
    else:
        mid_arr = np.full(n, float(mid))
    ts = 34_200_000 + 3000 * np.arange(n, dtype=np.int64)
    return DaySim(
        ts_ms=ts,
        bid1_px=np.full(n, bid),
        ask1_px=np.full(n, ask),
        mid_px=mid_arr,
        last_px=mid_arr.copy(),
        bid1_qty=np.full(n, bid_qty),
        ask1_qty=np.full(n, ask_qty),
        depth_bid5=np.full(n, 5 * bid_qty),
        depth_ask5=np.full(n, 5 * ask_qty),
        z=np.asarray(z, dtype=np.float64),
        tradable=(
            np.ones(n, dtype=bool) if tradable is None else np.asarray(tradable, bool)
        ),
    )


# --------------------------------------------------------------------------- #
# Execution: tick rounding, impact, price limits
# --------------------------------------------------------------------------- #
def test_fills_rounded_to_tick_grid() -> None:
    buy = cross_spread_fill("buy", 4.000, 4.001)
    sell = cross_spread_fill("sell", 4.000, 4.001)
    assert buy == pytest.approx(4.002)  # ask + 1 tick
    assert sell == pytest.approx(3.999)  # bid - 1 tick
    for px in (buy, sell):
        assert abs(px / TICK_CNY - round(px / TICK_CNY)) < 1e-9
    # half-tick slippage rounds conservatively onto the grid
    assert cross_spread_fill("buy", 4.000, 4.001, slippage_ticks=0.5) == pytest.approx(4.002)
    assert cross_spread_fill("sell", 4.000, 4.001, slippage_ticks=0.5) == pytest.approx(3.999)


def test_fill_invalid_book_returns_nan() -> None:
    assert np.isnan(cross_spread_fill("buy", 0.0, 4.001))
    assert np.isnan(cross_spread_fill("buy", 4.002, 4.001))  # crossed
    assert np.isnan(cross_spread_fill("sell", float("nan"), 4.001))
    with pytest.raises(ValueError):
        cross_spread_fill("hold", 4.0, 4.001)
    with pytest.raises(ValueError):
        cross_spread_fill("buy", 4.0, 4.001, slippage_ticks=-1.0)


def test_depth_impact_threshold_and_cap() -> None:
    assert depth_impact_bps(100.0, 10_000.0) == 0.0  # 1% participation <= 10%
    assert depth_impact_bps(1_000.0, 10_000.0) == 0.0  # exactly 10%: free
    assert depth_impact_bps(1_100.0, 10_000.0) == pytest.approx(0.05)  # kappa*(0.11-0.10)
    assert depth_impact_bps(1e9, 1e3) == pytest.approx(50.0)  # capped
    assert depth_impact_bps(100.0, 0.0) == pytest.approx(50.0)  # hollow book
    assert depth_impact_bps(0.0, 10_000.0) == 0.0


def test_price_limit_clamp() -> None:
    assert clamp_price_limit(11.5, 10.0) == pytest.approx(11.0)
    assert clamp_price_limit(8.5, 10.0) == pytest.approx(9.0)
    assert clamp_price_limit(10.5, 10.0) == pytest.approx(10.5)
    # no prior close: only tick rounding
    assert clamp_price_limit(10.5004, 0.0) == pytest.approx(10.5)
    assert np.isnan(clamp_price_limit(float("nan"), 10.0))


# --------------------------------------------------------------------------- #
# Signals: causal z-score and hysteresis positions
# --------------------------------------------------------------------------- #
def test_causal_zscore_warmup_and_values() -> None:
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 5.0])
    z = causal_zscore(x, 3)
    assert np.isnan(z[0]) and np.isnan(z[1])
    # window [1,2,3]: mean 2, population std sqrt(2/3)
    assert z[2] == pytest.approx((3.0 - 2.0) / np.sqrt(2.0 / 3.0))
    # trailing window only: z[3] uses [2,3,2]
    assert z[3] == pytest.approx((2.0 - 7.0 / 3.0) / np.std([2.0, 3.0, 2.0]))


def test_causal_zscore_nan_windows_and_edges() -> None:
    assert np.isnan(causal_zscore(np.array([1.0, 2.0]), 5)).all()
    with pytest.raises(ValueError):
        causal_zscore(np.array([1.0, 2.0, 3.0]), 1)
    # NaN inside the window is ignored; constant window -> neutral 0
    z = causal_zscore(np.array([1.0, np.nan, 3.0]), 3)
    assert z[2] == pytest.approx(1.0)  # (3 - 2) / 1
    z0 = causal_zscore(np.array([2.0, 2.0, 2.0, 2.0]), 3)
    assert z0[2] == 0.0 and z0[3] == 0.0
    # all-NaN window stays NaN
    zn = causal_zscore(np.array([np.nan, np.nan, np.nan]), 3)
    assert np.isnan(zn[2])


def test_position_hysteresis_lag_and_tradable() -> None:
    rule = PositionRule(entry_z=2.0, exit_z=0.5, max_position_units=1000)
    z = np.array([3.0, np.nan, np.nan, 3.0, 3.0, 0.0])
    tradable = np.array([True, True, False, True, True, True])
    pos = position_from_z(z, rule, tradable)
    assert pos[0] == 0.0  # lag: no decision possible at row 0
    assert pos[1] == pytest.approx(1000.0)  # entry from z[0]
    assert pos[2] == 0.0  # untradable row forces flat
    assert pos[3] == 0.0  # state was reset; z[2] is NaN anyway
    assert pos[4] == pytest.approx(1000.0)  # fresh entry from z[3]
    # decision at row 5 uses z[4]=3 -> still holding (exit needs |z| < 0.5)
    assert pos[5] == pytest.approx(1000.0)


def test_position_rule_validation() -> None:
    with pytest.raises(ValueError):
        position_from_z(np.zeros(3), PositionRule(signal_lag_rows=0), np.ones(3, bool))
    with pytest.raises(ValueError):
        position_from_z(np.zeros(3), PositionRule(entry_z=1.0, exit_z=1.5), np.ones(3, bool))
    with pytest.raises(ValueError):
        position_from_z(np.zeros(3), PositionRule(direction=0), np.ones(3, bool))


def test_position_direction_inverts_signal() -> None:
    z = np.array([np.nan, -3.0, -3.0, -3.0])
    pos = position_from_z(z, PositionRule(direction=-1, max_position_units=1000), np.ones(4, bool))
    assert pos[2] == pytest.approx(1000.0)  # negative z + direction -1 => long


def test_zscore_column_groups_and_preserves_order() -> None:
    df = pl.DataFrame(
        {
            "instrument": ["A"] * 4 + ["B"] * 4,
            "date": ["20250603"] * 8,
            "f": [0.0, 0.0, 0.0, 1.0, 5.0, 5.0, 5.0, 9.0],
        }
    )
    out = zscore_column(df, "f", window_rows=3)
    assert "f_z" in out.columns
    assert out.height == 8
    z_a = out.filter(pl.col("instrument") == "A")["f_z"].to_numpy()
    assert np.isnan(z_a[0]) and np.isnan(z_a[1])
    assert z_a[2] == 0.0  # constant window
    # window [0,0,1]: mean 1/3, std sqrt(2/9) -> z = (2/3)/sqrt(2/9) = sqrt(2)
    assert z_a[3] == pytest.approx(2.0 ** 0.5)
    # group isolation: B's first window does not include A's rows
    z_b = out.filter(pl.col("instrument") == "B")["f_z"].to_numpy()
    assert np.isnan(z_b[0]) and np.isnan(z_b[1])


# --------------------------------------------------------------------------- #
# simulate_day: PnL arithmetic, fees, impact overlay
# --------------------------------------------------------------------------- #
def test_pnl_arithmetic_flat_to_long_mark_to_market() -> None:
    z = [3.0] * 6
    mids = [4.0005] + [4.010] * 5
    day = make_day(n=6, z=z, mid=mids)
    res = simulate_day(day, T0_META, ZERO_COSTS, RULE, sellable_start_units=0.0)

    assert res.trades_units[1] == pytest.approx(1000.0)  # entry at row 1 (lag 1)
    assert res.position[0] == 0.0 and res.position[1] == pytest.approx(1000.0)
    # buy 1000 @ 4.002 (ask 4.001 + 1 tick), mark ends at 4.010
    assert res.cash_pnl.sum() == pytest.approx(1000.0 * (4.010 - 4.002))
    assert res.fees_cny == 0.0
    assert res.sellable_end_units == pytest.approx(1000.0)


def test_fees_accounted_per_order() -> None:
    costs = CostModel(
        name="c", commission_rate=0.0002, min_commission_cny=0.0,
        handling_fee_rate=0.0, regulatory_fee_rate=0.0, transfer_fee_rate=0.0,
    )
    z = [3.0] * 6
    day = make_day(n=6, z=z)
    res = simulate_day(day, T0_META, costs, RULE, sellable_start_units=0.0)
    # one buy order: 0.0002 * 1000 units * 4.002 fill
    assert res.fees_cny == pytest.approx(0.0002 * 1000.0 * 4.002)


def test_depth_impact_overlay_moves_the_fill() -> None:
    z = [3.0] * 6
    # ask-side depth only 1000 units -> participation ~100% -> +4.5bp impact
    day = make_day(n=6, z=z, ask_qty=1000.0)
    res = simulate_day(day, T0_META, ZERO_COSTS, RULE, sellable_start_units=0.0)
    # fill = 4.002 * (1 + 5*(1.00025-0.10)/1e4) = 4.0038 -> ceil tick 4.004
    assert res.slippage_cost_cny == pytest.approx(1000.0 * (4.004 - 4.0005))
    assert res.cash_pnl.sum() == pytest.approx(1000.0 * (4.0005 - 4.004))

    # with the overlay disabled the fill stays at the crossed ask + slippage
    res_off = simulate_day(
        day, T0_META, ZERO_COSTS, RULE,
        sellable_start_units=0.0, use_depth_impact=False,
    )
    assert res_off.cash_pnl.sum() == pytest.approx(1000.0 * (4.0005 - 4.002))


def test_no_trading_without_signal_or_when_untradable() -> None:
    day = make_day(n=6, z=[0.0] * 6)
    res = simulate_day(day, T0_META, ZERO_COSTS, RULE, sellable_start_units=0.0)
    assert np.count_nonzero(res.trades_units) == 0
    assert res.fees_cny == 0.0

    day2 = make_day(n=6, z=[3.0] * 6, tradable=[False] * 6)
    res2 = simulate_day(day2, T0_META, ZERO_COSTS, RULE, sellable_start_units=0.0)
    assert np.count_nonzero(res2.trades_units) == 0


def test_position_force_flat_before_intraday_untradable_gap() -> None:
    # Rows 4..7 are untradable (lunch/flagged gap): the position open from
    # row 1 must be liquidated on row 3, the last tradable row before the
    # gap, instead of being carried across it.
    tradable = [True, True, True, True, False, False, False, False]
    day = make_day(n=8, z=[3.0] * 8, tradable=tradable)
    res = simulate_day(day, T0_META, ZERO_COSTS, RULE, sellable_start_units=0.0)
    assert res.trades_units[1] == pytest.approx(1000.0)   # entry
    assert res.trades_units[3] == pytest.approx(-1000.0)  # forced flatten
    assert np.count_nonzero(res.trades_units) == 2
    assert res.sellable_end_units == 0.0
    assert res.position[-1] == 0.0


def test_end_of_day_position_not_force_flattened() -> None:
    # All rows tradable: a position held to the final row is carried out as
    # tomorrow's sellable pool (底仓 chaining), NOT liquidated at the close.
    day = make_day(n=6, z=[3.0] * 6)
    res = simulate_day(day, T0_META, ZERO_COSTS, RULE, sellable_start_units=0.0)
    assert res.trades_units[1] == pytest.approx(1000.0)
    assert np.count_nonzero(res.trades_units) == 1
    assert res.sellable_end_units == pytest.approx(1000.0)


def test_odd_lot_full_liquidation_allowed() -> None:
    # Inventory of 150 units (one odd lot) must be sellable in one order.
    # The sell lands on row 1, the first row with decision input.
    day = make_day(n=6, z=[0.0] * 6)
    res = simulate_day(day, T0_META, ZERO_COSTS, RULE, sellable_start_units=150.0)
    assert res.trades_units[0] == 0.0
    assert res.trades_units[1] == pytest.approx(-150.0)
    assert res.sellable_end_units == 0.0


def test_tick_meta_mismatch_rejected() -> None:
    bad = InstrumentMeta(exchange="sse", settlement="T+0", tick_cny=0.01)
    day = make_day(n=6, z=[0.0] * 6)
    with pytest.raises(ValueError):
        simulate_day(day, bad, ZERO_COSTS, RULE)


def test_position_size_vs_order_limits_enforced() -> None:
    day = make_day(n=6, z=[3.0] * 6)
    too_big = PositionRule(max_position_units=2_000_000)  # > 1M max order
    with pytest.raises(ValueError):
        simulate_day(day, T0_META, ZERO_COSTS, too_big)
    odd_lot = PositionRule(max_position_units=150)  # not a multiple of lot 100
    with pytest.raises(ValueError):
        simulate_day(day, T0_META, ZERO_COSTS, odd_lot)


# --------------------------------------------------------------------------- #
# run_backtest: session exclusion and end-to-end behaviour
# --------------------------------------------------------------------------- #
def _panel_row(date, exch, inst, ts, factor, flags=0):
    return {
        "date": date,
        "exchange": exch,
        "instrument": inst,
        "ts_ms": ts,
        "flags": flags,
        "mid_px": 4.0005,
        "last_px": 4.0005,
        "bid1_px": 4.000,
        "ask1_px": 4.001,
        "bid1_qty": 1e9,
        "ask1_qty": 1e9,
        "depth_bid5": 5e9,
        "depth_ask5": 5e9,
        "oir": factor,
    }


def test_session_and_auction_rows_never_traded() -> None:
    rows = []
    # SSE instrument: rows in-session (09:30:00..09:30:30) plus rows outside
    # the continuous session (11:31:00 lunch, 15:00:03 after close) carrying
    # huge factor spikes that would trade if they leaked in.
    for i in range(11):
        rows.append(_panel_row("20250603", "sse", "510300", 34_200_000 + 3000 * i, 0.0))
    rows.append(_panel_row("20250603", "sse", "510300", 41_460_000, 100.0))  # 11:31
    rows.append(_panel_row("20250603", "sse", "510300", 54_003_000, 100.0))  # 15:00:03
    # SZSE instrument: in-session rows at 13:00 plus one inside the 14:57-15:00
    # closing auction, which must be excluded.
    for i in range(11):
        rows.append(_panel_row("20250603", "szse", "159915", 46_800_000 + 3000 * i, 0.0))
    rows.append(_panel_row("20250603", "szse", "159915", 53_880_000, 100.0))  # 14:58

    panel = pl.DataFrame(rows)
    res = run_backtest(
        panel, "oir", 15, {}, ZERO_COSTS, RULE, dates=["20250603"], z_window_rows=5
    )
    assert res.n_trades == 0
    assert res.total_pnl_cny == 0.0
    assert res.per_day.height == 2  # both instruments had in-session rows
    assert res.per_day["position_end_units"].to_list() == [0.0, 0.0]


def test_flagged_and_one_sided_rows_untradable() -> None:
    rows = []
    for i in range(12):
        rows.append(_panel_row("20250603", "sse", "510300", 34_200_000 + 3000 * i, 0.0))
    panel = pl.DataFrame(rows)
    # book-unsynced flag on the entry rows kills the entry entirely
    flagged = panel.with_columns(
        pl.when(pl.col("ts_ms").is_in([34_203_000, 34_206_000]))
        .then(1)
        .otherwise(pl.col("flags"))
        .alias("flags")
    )
    res_flag = run_backtest(
        flagged, "oir", 15, {}, ZERO_COSTS, RULE, dates=["20250603"], z_window_rows=5
    )
    res_clean = run_backtest(
        panel, "oir", 15, {}, ZERO_COSTS, RULE, dates=["20250603"], z_window_rows=5
    )
    assert res_flag.n_trades == res_clean.n_trades == 0


def test_run_backtest_end_to_end_entry_exit_and_costs() -> None:
    n = 32
    ts = [34_200_000 + 3000 * i for i in range(n)]
    # z hits exactly +2.0 at the jump (window [0,0,0,0,1]) then decays to 0
    # once the window is all-ones; the extra tail rows give the exit decision
    # (actuated with the 1-row lag) room to land inside the day.
    factor = [0.0] * 25 + [1.0] * 7
    rows = [
        _panel_row("20250603", "sse", "511880", ts[i], factor[i]) for i in range(n)
    ]
    panel = pl.DataFrame(rows)
    rule = PositionRule(entry_z=2.0, exit_z=0.5, max_position_units=10_000)
    meta = {"511880": T0_META}

    res = run_backtest(panel, "oir", 15, meta, ZERO_COSTS, rule, z_window_rows=5)
    # buy at row 26 (decision on z[25]=2.0), exit sell at row 30 (z[29]=0)
    assert res.n_trades == 2  # one buy, one sell
    # buy @ ask+1 tick = 4.002, sell @ bid-1 tick = 3.999 -> -0.003 per unit
    assert res.total_pnl_cny == pytest.approx(-0.003 * 10_000)
    assert res.total_fees_cny == 0.0
    assert res.realized_round_trip_cost_bps > 0  # slippage still costs
    assert res.n_days == 1
    assert res.sharpe_annualized == 0.0  # one day -> undefined -> 0
    assert set(res.per_day.columns) >= {
        "date", "instrument", "pnl_cny", "fees_cny", "n_trades", "t1_blocked_units",
    }
    assert res.equity_curve.height == 1
    assert res.per_day["position_end_units"][0] == 0.0

    # same day as a T+1 equity ETF: the closing sell is blocked
    res_t1 = run_backtest(
        panel, "oir", 15, {"511880": T1_META}, ZERO_COSTS, rule, z_window_rows=5
    )
    assert res_t1.per_day["t1_blocked_units"][0] == pytest.approx(10_000.0)
    assert res_t1.per_day["position_end_units"][0] == pytest.approx(10_000.0)


def test_run_backtest_missing_columns_raise() -> None:
    panel = pl.DataFrame({"date": ["20250603"], "instrument": ["510300"]})
    with pytest.raises(ValueError):
        run_backtest(panel, "oir", 15, {}, ZERO_COSTS, RULE)


def test_run_backtest_empty_after_date_filter() -> None:
    rows = [_panel_row("20250603", "sse", "510300", 34_200_000, 0.0)]
    panel = pl.DataFrame(rows)
    res = run_backtest(panel, "oir", 15, {}, ZERO_COSTS, RULE, dates=["20990101"])
    assert res.n_days == 0 and res.n_trades == 0
    assert res.per_day.height == 0 and res.equity_curve.height == 0


# --------------------------------------------------------------------------- #
# Metrics: summary table and the multi-scenario gate
# --------------------------------------------------------------------------- #
def _mk_result(
    pnl: float, fees: float, sharpe: float, n_days: int
) -> BacktestResult:
    dates = [f"202506{d:02d}" for d in range(1, n_days + 1)]
    per_day = pl.DataFrame(
        {
            "date": dates,
            "instrument": ["510300"] * n_days,
            "pnl_cny": [pnl / n_days] * n_days,
            "fees_cny": [fees / n_days] * n_days,
            "n_trades": [2] * n_days,
            "t1_blocked_units": [0.0] * n_days,
        }
    )
    eq = pl.DataFrame(
        {
            "date": dates,
            "pnl_cny": [pnl / n_days] * n_days,
            "equity_cny": [(i + 1) * pnl / n_days for i in range(n_days)],
            "drawdown_cny": [0.0] * n_days,
        }
    )
    return BacktestResult(
        per_day=per_day,
        equity_curve=eq,
        total_pnl_cny=pnl,
        total_fees_cny=fees,
        sharpe_annualized=sharpe,
        max_drawdown_cny=0.0,
        turnover_units_per_day=1000.0,
        realized_round_trip_cost_bps=2.0,
        n_days=n_days,
        n_trades=2,
    )


def test_summarize_results_one_row_per_scenario() -> None:
    results = {
        "institutional": _mk_result(100.0, 5.0, 2.0, 30),
        "retail_default": _mk_result(-50.0, 40.0, -0.5, 30),
    }
    s = summarize_results(results)
    assert s.height == 2
    assert set(s.columns) >= {
        "scenario", "n_days", "total_pnl_cny", "total_fees_cny", "sharpe_annualized",
        "capacity_proxy_cny",
    }
    assert set(s["scenario"].to_list()) == {"institutional", "retail_default"}
    # no traded_notional_cny column in the synthetic per_day frames -> 0
    assert (s["capacity_proxy_cny"].to_numpy() == 0.0).all()


def test_summarize_capacity_proxy_from_per_day() -> None:
    r = _mk_result(100.0, 5.0, 2.0, 3)
    r.per_day = r.per_day.with_columns(
        pl.Series("traded_notional_cny", [1.0e6, 4.0e6, 2.5e6])
    )
    s = summarize_results({"institutional": r})
    assert s["capacity_proxy_cny"][0] == pytest.approx(4.0e6)


def test_gate_requires_survival_in_all_scenarios() -> None:
    good = _mk_result(100.0, 5.0, 2.0, 30)
    weak_sharpe = _mk_result(10.0, 5.0, 0.3, 30)
    ok, details = gate_on_costs({"institutional": good, "retail_default": good})
    assert ok is True and details["passed"] is True

    ok2, details2 = gate_on_costs(
        {"institutional": good, "retail_default": weak_sharpe}
    )
    assert ok2 is False
    assert details2["scenarios"]["retail_default"]["pass"] is False
    assert details2["scenarios"]["institutional"]["pass"] is True


def test_gate_min_days_and_empty_inputs() -> None:
    short = _mk_result(100.0, 5.0, 3.0, 5)
    ok, details = gate_on_costs({"institutional": short}, min_days=20)
    assert ok is False
    assert any("days" in r for r in details["scenarios"]["institutional"]["reasons"])

    ok_empty, details_empty = gate_on_costs({})
    assert ok_empty is False and details_empty["scenarios"] == {}


def test_gate_negative_pnl_fails_even_with_sharpe() -> None:
    neg = _mk_result(-1.0, 5.0, 3.0, 30)
    ok, details = gate_on_costs({"institutional": neg})
    assert ok is False
    assert any("PnL" in r for r in details["scenarios"]["institutional"]["reasons"])
