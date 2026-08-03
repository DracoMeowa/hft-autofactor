"""T+1 sell-lock tests: same-day buys unsellable, cross-day inventory chaining,
and T+0 categories unaffected by the lock.

The engine enforces ``sellable_qty(t) = holdings(t-1)`` for equity ETFs
(SSE trading rules 3.1.4 / ETF detail rules art.22): units bought today can
be redeemed but NOT sold today, so intraday round trips require pre-held
inventory (底仓).  Bond/money/gold/commodity/cross-border ETFs are T+0 and
may sell same-day buys.
"""
from __future__ import annotations

import numpy as np
import pytest

from hft_autofactor.backtest.costs import CostModel
from hft_autofactor.backtest.engine import DaySim, InstrumentMeta, simulate_day
from hft_autofactor.backtest.signals import PositionRule

#: Zero-cost model to isolate settlement mechanics from the fee stack.
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


def make_day(
    n: int = 12,
    *,
    z: list[float],
    bid: float = 4.000,
    ask: float = 4.001,
    qty: float = 1e9,
) -> DaySim:
    mid = 0.5 * (bid + ask)
    ts = 34_200_000 + 3000 * np.arange(n, dtype=np.int64)  # from 09:30:00
    return DaySim(
        ts_ms=ts,
        bid1_px=np.full(n, bid),
        ask1_px=np.full(n, ask),
        mid_px=np.full(n, mid),
        last_px=np.full(n, mid),
        bid1_qty=np.full(n, qty),
        ask1_qty=np.full(n, qty),
        depth_bid5=np.full(n, 5 * qty),
        depth_ask5=np.full(n, 5 * qty),
        z=np.asarray(z, dtype=np.float64),
        tradable=np.ones(n, dtype=bool),
    )


# z pattern: enter long at row 2 (lag 1 of z[1]=3), attempt exit at row 5
# (lag 1 of z[4]=0 -> |z| < exit_z).
ENTRY_EXIT_Z = [float("nan"), 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

META_T1 = InstrumentMeta(exchange="sse", settlement="T+1", etf_category="equity_etf")
META_T0 = InstrumentMeta(exchange="sse", settlement="T+0", etf_category="bond_etf")


def test_t1_same_day_buy_cannot_be_sold() -> None:
    day = make_day(z=ENTRY_EXIT_Z)
    res = simulate_day(day, META_T1, ZERO_COSTS, RULE, sellable_start_units=0.0)

    # Buy executes on the entry row (row 2).
    assert res.trades_units[2] == pytest.approx(1000.0)
    # The exit sell at row 5 is blocked by the T+1 lock (sellable pool is 0).
    assert res.trades_units[5] == 0.0
    assert res.t1_blocked_units == pytest.approx(1000.0)
    # Position stays long through the close; holdings become tomorrow's pool.
    assert res.position[-1] == pytest.approx(1000.0)
    assert res.sellable_end_units == pytest.approx(1000.0)
    # No other trades happened.
    assert np.count_nonzero(res.trades_units) == 1


def test_t1_blocked_sell_not_retried_every_row() -> None:
    # If the blocked shortfall were re-counted on every row, blocked units
    # would be ~7x larger; the lock demand is counted once per target change.
    day = make_day(z=ENTRY_EXIT_Z)
    res = simulate_day(day, META_T1, ZERO_COSTS, RULE, sellable_start_units=0.0)
    assert res.t1_blocked_units == pytest.approx(1000.0)


def test_t1_inventory_chainable_and_sellable_next_day() -> None:
    # Day 2 starts with yesterday's closing holdings (all now sellable) and a
    # flat signal -> the position liquidates on the first row with decision
    # input (row 1; row 0 has no lagged z yet) and does so legally.
    z_flat = [0.0] * 12
    day2 = make_day(z=z_flat)
    res = simulate_day(day2, META_T1, ZERO_COSTS, RULE, sellable_start_units=1000.0)

    assert res.trades_units[0] == 0.0  # no decision input at row 0 yet
    assert res.trades_units[1] == pytest.approx(-1000.0)
    assert res.t1_blocked_units == 0.0
    assert res.sellable_end_units == 0.0
    assert res.position[-1] == 0.0


def test_t1_two_day_chain_via_sellable_end() -> None:
    # Full chain: day 1 buys (locked), day 2 sells exactly the carried units.
    day1 = make_day(z=ENTRY_EXIT_Z)
    r1 = simulate_day(day1, META_T1, ZERO_COSTS, RULE, sellable_start_units=0.0)
    assert r1.sellable_end_units == pytest.approx(1000.0)

    day2 = make_day(z=[0.0] * 12)
    r2 = simulate_day(
        day2, META_T1, ZERO_COSTS, RULE, sellable_start_units=r1.sellable_end_units
    )
    assert r2.t1_blocked_units == 0.0
    assert r2.sellable_end_units == 0.0


def test_t0_same_day_round_trip_allowed() -> None:
    day = make_day(z=ENTRY_EXIT_Z)
    res = simulate_day(day, META_T0, ZERO_COSTS, RULE, sellable_start_units=0.0)

    assert res.trades_units[2] == pytest.approx(1000.0)  # buy
    assert res.trades_units[5] == pytest.approx(-1000.0)  # same-day sell OK
    assert res.t1_blocked_units == 0.0
    assert res.sellable_end_units == 0.0
    assert res.position[-1] == 0.0


def test_t0_sellable_pool_grows_with_buys() -> None:
    # Enter, exit, re-enter, exit again: every same-day sell must be allowed.
    z = [float("nan"), 3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    day = make_day(z=z)
    res = simulate_day(day, META_T0, ZERO_COSTS, RULE, sellable_start_units=0.0)
    assert res.t1_blocked_units == 0.0
    assert res.sellable_end_units == 0.0
    # two buys + two sells
    assert np.count_nonzero(res.trades_units) == 4


def test_partial_sellable_pool_allows_partial_sell() -> None:
    # Carry 1000 sellable units; signal expands the position to 2000 (buying
    # 1000 same-day units) and then exits.  Only the carried 1000 may be
    # sold; today's 1000 bought units stay locked until tomorrow.
    rule2k = PositionRule(
        entry_z=2.0, exit_z=0.5, max_position_units=2000, signal_lag_rows=1
    )
    z = [3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    day = make_day(z=z)
    day.tradable[0] = False  # first snapshot untradable -> no open-auction sell
    res = simulate_day(day, META_T1, ZERO_COSTS, rule2k, sellable_start_units=1000.0)

    # row 1: entry to 2000 -> buy 1000 on top of the carried 1000
    assert res.trades_units[1] == pytest.approx(1000.0)
    # row 5: exit -> demand 2000, allowed only the 1000 carried sellable units
    assert res.trades_units[5] == pytest.approx(-1000.0)
    assert res.t1_blocked_units == pytest.approx(1000.0)
    assert res.sellable_end_units == pytest.approx(1000.0)  # today's buy locked
    assert res.position[-1] == pytest.approx(1000.0)


def test_negative_inventory_rejected() -> None:
    day = make_day(z=[0.0] * 12)
    with pytest.raises(ValueError):
        simulate_day(day, META_T1, ZERO_COSTS, RULE, sellable_start_units=-1.0)


def test_t1_force_flat_before_gap_only_sells_sellable_pool() -> None:
    # Same-day units bought before an intraday untradable gap cannot be
    # flattened there under T+1: only the carried sellable pool may sell;
    # the locked remainder is carried (and chains into the next day).
    z = [3.0] * 12
    day = make_day(z=z)
    day.tradable[5:] = False  # gap starts at row 5
    res = simulate_day(
        day, META_T1, ZERO_COSTS, RULE, sellable_start_units=1000.0
    )
    # row 1: entry target 1000 -> holdings already 1000 carried, no trade...
    # max_position_units=1000 means target == carried inventory: nothing to do
    assert res.trades_units[1] == 0.0
    # row 4: forced flatten before the gap sells exactly the carried pool
    assert res.trades_units[4] == pytest.approx(-1000.0)
    assert res.t1_blocked_units == 0.0
    assert res.sellable_end_units == 0.0

    # With a bigger target, same-day buys stay locked through the gap.
    rule2k = PositionRule(
        entry_z=2.0, exit_z=0.5, max_position_units=2000, signal_lag_rows=1
    )
    day2 = make_day(z=z)
    day2.tradable[5:] = False
    res2 = simulate_day(
        day2, META_T1, ZERO_COSTS, rule2k, sellable_start_units=1000.0
    )
    assert res2.trades_units[1] == pytest.approx(1000.0)   # buy on top
    assert res2.trades_units[4] == pytest.approx(-1000.0)  # sell carried only
    assert res2.t1_blocked_units == pytest.approx(1000.0)  # today's buy locked
    assert res2.sellable_end_units == pytest.approx(1000.0)
