"""Base-inventory (底仓) rule tests: targets around the inventory floor and
T+1 intraday round trips (底仓做T).

With ``base_units > 0`` the hysteresis targets become
``base_units + state * (max_position_units - base_units)`` (state -1/0/+1),
so an exit returns to the floor instead of flatting, and the engine's
forced flatten before intraday untradable gaps also targets the floor.
Combined with an opening T+1 sellable pool equal to the base, this is the
classic A-share construction that makes intraday round trips possible on
T+1 equity ETFs: sells draw down the carried pool while same-day buys stay
locked until tomorrow.
"""
from __future__ import annotations

import numpy as np
import pytest

from hft_autofactor.backtest.costs import CostModel
from hft_autofactor.backtest.engine import DaySim, InstrumentMeta, simulate_day
from hft_autofactor.backtest.signals import PositionRule, position_from_z

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

META_T1 = InstrumentMeta(exchange="sse", settlement="T+1", etf_category="equity_etf")


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


def base_rule(base: int, max_units: int) -> PositionRule:
    return PositionRule(
        entry_z=2.0,
        exit_z=0.5,
        direction=1,
        max_position_units=max_units,
        signal_lag_rows=1,
        base_units=base,
    )


# --------------------------------------------------------------------------
# position_from_z target mapping
# --------------------------------------------------------------------------


def test_targets_oscillate_around_base_floor() -> None:
    rule = base_rule(base=500, max_units=1500)
    # decisions at row i consume z[i-1] (lag 1)
    z = np.array([3.0, 3.0, -3.0, 0.2, 0.0])
    tradable = np.ones(5, dtype=bool)
    pos = position_from_z(z, rule, tradable)
    # row 0: no decision input yet -> floor
    # rows 1-2: same-side band -> max
    # row 3: opposite band -> 2*base - max = -500 (engine clips to 0)
    # row 4: |z| < exit -> back to the floor
    assert pos.tolist() == pytest.approx([500.0, 1500.0, 1500.0, -500.0, 500.0])


def test_base_zero_reproduces_plain_long_flat() -> None:
    rule = base_rule(base=0, max_units=1500)
    z = np.array([3.0, 3.0, -3.0, 0.2, 0.0])
    tradable = np.ones(5, dtype=bool)
    pos = position_from_z(z, rule, tradable)
    # Identical to the pre-base-floor mapping: 0 (flat/no signal), max on the
    # same-side band, -max on the opposite band (engines clip it to 0).
    assert pos.tolist() == pytest.approx([0.0, 1500.0, 1500.0, -1500.0, 0.0])


def test_untradable_rows_force_zero_and_reset_with_base() -> None:
    rule = base_rule(base=500, max_units=1500)
    z = np.array([3.0, 3.0, 1.0, 1.0])
    tradable = np.array([True, True, False, True])
    pos = position_from_z(z, rule, tradable)
    assert pos[1] == pytest.approx(1500.0)
    assert pos[2] == 0.0  # untradable rows carry no target
    # State was reset by the untradable row; the decision input at row 3 is
    # z[2]=1.0, below the entry band, so no fresh entry -> back to the floor
    # (the stale band at row 1 is NOT carried across the gap).
    assert pos[3] == pytest.approx(500.0)


def test_base_units_out_of_range_rejected() -> None:
    z = np.zeros(3)
    tradable = np.ones(3, dtype=bool)
    with pytest.raises(ValueError):
        position_from_z(z, base_rule(base=2000, max_units=1500), tradable)
    with pytest.raises(ValueError):
        position_from_z(z, base_rule(base=-100, max_units=1500), tradable)


# --------------------------------------------------------------------------
# T+1 intraday round trips funded by the carried sellable pool
# --------------------------------------------------------------------------


ENTRY_EXIT_Z = [float("nan"), 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_buy_first_cycle_completes_intraday_under_t1() -> None:
    # Base 1000 / max 2000: entry buys the 1000 deviation, exit sells it
    # back OUT OF THE CARRIED POOL (today's buy stays locked) -> a complete
    # intraday round trip despite T+1.
    day = make_day(z=ENTRY_EXIT_Z)
    res = simulate_day(
        day, META_T1, ZERO_COSTS, base_rule(1000, 2000), sellable_start_units=1000.0
    )
    assert res.trades_units[2] == pytest.approx(1000.0)   # buy deviation
    assert res.trades_units[5] == pytest.approx(-1000.0)  # sell from pool
    assert res.t1_blocked_units == 0.0
    assert res.position[-1] == pytest.approx(1000.0)      # back on the floor
    assert res.sellable_end_units == pytest.approx(1000.0)


def test_sell_first_cycle_completes_intraday_under_t1() -> None:
    # Negative entry sells down to 2*base - max = 0, exit buys back to base.
    z = [float("nan"), -3.0, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    day = make_day(z=z)
    res = simulate_day(
        day, META_T1, ZERO_COSTS, base_rule(1000, 2000), sellable_start_units=1000.0
    )
    assert res.trades_units[2] == pytest.approx(-1000.0)  # sell from pool
    assert res.trades_units[5] == pytest.approx(1000.0)   # buy back to base
    assert res.t1_blocked_units == 0.0
    assert res.position[-1] == pytest.approx(1000.0)
    assert res.sellable_end_units == pytest.approx(1000.0)


def test_sellable_pool_depletes_one_deviation_per_cycle() -> None:
    # base 1000 / max 1500 (deviation 500): two sell cycles drain the pool,
    # the third sell attempt is T+1-blocked and not retried.
    z = [float("nan"), -3.0, -3.0, 0.0, 0.0, -3.0, -3.0, 0.0, 0.0, -3.0, -3.0, 0.0]
    day = make_day(z=z)
    res = simulate_day(
        day, META_T1, ZERO_COSTS, base_rule(1000, 1500), sellable_start_units=1000.0
    )
    sells = res.trades_units[res.trades_units < 0]
    buys = res.trades_units[res.trades_units > 0]
    assert len(sells) == 2  # only two sells funded by the pool
    assert len(buys) == 2   # both exits buy back to the floor
    assert res.t1_blocked_units == pytest.approx(500.0)  # third sell demand
    assert res.position[-1] == pytest.approx(1000.0)     # back on the floor


def test_force_flat_before_gap_targets_base_not_zero() -> None:
    # Long at max when an intraday gap starts: the forced flatten sells only
    # the deviation and carries the base inventory through the gap.
    z = [3.0] * 12
    day = make_day(z=z)
    day.tradable[5:] = False
    res = simulate_day(
        day, META_T1, ZERO_COSTS, base_rule(1000, 2000), sellable_start_units=1000.0
    )
    assert res.trades_units[1] == pytest.approx(1000.0)   # buy deviation
    assert res.trades_units[4] == pytest.approx(-1000.0)  # sell back to base
    assert res.t1_blocked_units == 0.0
    assert res.position[-1] == pytest.approx(1000.0)      # base carried
    assert res.sellable_end_units == pytest.approx(1000.0)


def test_no_signal_day_keeps_base_inventory() -> None:
    # z identically 0: never reaches the entry band -> no trade at all and
    # the base inventory chains untouched.
    day = make_day(z=[0.0] * 12)
    res = simulate_day(
        day, META_T1, ZERO_COSTS, base_rule(1000, 2000), sellable_start_units=1000.0
    )
    assert np.count_nonzero(res.trades_units) == 0
    assert res.sellable_end_units == pytest.approx(1000.0)
    assert res.position[-1] == pytest.approx(1000.0)


def test_base_units_must_be_lot_multiple() -> None:
    day = make_day(z=[0.0] * 12)
    with pytest.raises(ValueError):
        simulate_day(
            day, META_T1, ZERO_COSTS, base_rule(50, 1000), sellable_start_units=50.0
        )
