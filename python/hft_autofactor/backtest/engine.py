"""Vectorized per-day backtest engine with A-share ETF settlement rules.

Design notes
------------
Session calendar (mirrors cpp/hftaf/session.hpp and fee_table_v1):

* SSE funds trade continuously 09:30-11:30 / 13:00-15:00 with NO closing
  auction (close = last-1-min VWAP, trading rules 4.1.3).
* SZSE continuous trading ends 14:57; the 14:57-15:00 closing auction is
  excluded from the backtest grid.
* Call auctions (09:15-09:25) are never part of the grid -- factor rows do
  not exist there.

T+1 settlement (底仓 model)
---------------------------
Equity ETFs are T+1: units bought today CANNOT be sold today (SSE trading
rules 3.1.4).  The engine enforces ``sellable_qty(t) = holdings(t-1)``:
``simulate_day`` receives ``sellable_start_units`` (units carried from the
previous day, all of which are sellable) and sells are bounded by the
sellable pool; same-day buys never enter it.  T+0 categories (bond / money /
gold / commodity / commodity-futures / cross-border ETFs, per
etf_backtest_params.yaml ``settlement.t_plus_0``) may sell same-day buys.
``run_backtest`` chains ``sellable_end_units`` of one day into
``sellable_start_units`` of the next, per instrument, with optional initial
inventory (``initial_inventory_units``).  Naive intraday buy-then-sell round
trips for equity ETFs are therefore impossible by construction.

Positions
---------
Targets come from :func:`backtest.signals.position_from_z` (hysteresis rule,
actuation lag >= 1 row) and are clipped to ``[0, max_position_units]``: there
is no spot shorting of A-share ETFs in v1, so a negative target means flat.
Rows inside ``horizon_s`` of a continuous-session boundary (lunch or close)
are untradable: labels never span those boundaries, so no position may be
opened there.  A position still open on the last tradable row before an
intraday untradable gap (lunch break, flagged data interval) is force-flat
there (subject to the T+1 sellable pool); the day's FINAL row is exempt --
closing inventory chains into the next day as the sellable pool (底仓 model),
and the overnight mark-to-market gap is credited by ``run_backtest``.

PnL accounting
--------------
Per-row cash-flow PnL marks holdings to mid (fallback last):
``pnl_i = H_i*m_i - H_{i-1}*m_{i-1} - Δ_i*f_i`` (fees excluded; reported
separately).  ``run_backtest`` additionally credits the overnight gap on
carried inventory, ``H_start * (m_first_today - m_last_prev_day)``, so the
multi-day mark-to-market is complete.  Rows whose decision input (the z-score
``signal_lag_rows`` back) is absent -- pre-lag rows, warm-up, signal dropout
-- never trade, so inventory is not liquidated before the signal is
available.  Rows without a usable mark (one-sided book AND no last price)
contribute zero row PnL instead of poisoning the telescoping sum with NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import polars as pl

from .costs import CostModel, side_cost_cny
from .execution import (
    TICK_CNY,
    _ceil_tick,
    _floor_tick,
    cross_spread_fill,
    depth_impact_bps,
)
from .signals import PositionRule, causal_zscore, position_from_z

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "InstrumentMeta",
    "DaySim",
    "DayResult",
    "BacktestResult",
    "simulate_day",
    "run_backtest",
    "ANN_TRADING_DAYS",
]

#: A-share trading days per year used for Sharpe annualization.
ANN_TRADING_DAYS: int = 242

_EPS_UNITS = 1e-9

# Continuous-session bounds in ms since midnight, mirroring cpp session.hpp.
# SSE funds: no closing auction (continuous to 15:00).
# SZSE: continuous ends 14:57; 14:57-15:00 auction excluded.
_SESSIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "sse": ((34_200_000, 41_400_000), (46_800_000, 54_000_000)),
    "szse": ((34_200_000, 41_400_000), (46_800_000, 53_820_000)),
}

# flags bits that make a row untradable (book unsynced / SeqNo gap before).
_BAD_FLAG_MASK = 0b11

_PER_DAY_SCHEMA: dict[str, object] = {
    "date": pl.String,
    "instrument": pl.String,
    "pnl_cny": pl.Float64,
    "fees_cny": pl.Float64,
    "n_trades": pl.Int64,
    "t1_blocked_units": pl.Float64,
    "slippage_cost_cny": pl.Float64,
    "gap_pnl_cny": pl.Float64,
    "position_end_units": pl.Float64,
    "traded_units": pl.Float64,
    "traded_notional_cny": pl.Float64,
}

_EQUITY_SCHEMA: dict[str, object] = {
    "date": pl.String,
    "pnl_cny": pl.Float64,
    "equity_cny": pl.Float64,
    "drawdown_cny": pl.Float64,
}


def session_bounds(exchange: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Continuous-session bounds (ms since midnight) for an exchange."""
    key = exchange.lower()
    if key not in _SESSIONS:
        raise ValueError(f"unknown exchange {exchange!r} (expected 'sse' or 'szse')")
    return _SESSIONS[key]


def _session_mask(ts: np.ndarray, exchange: str) -> np.ndarray:
    mask = np.zeros(ts.shape, dtype=bool)
    for a, b in session_bounds(exchange):
        mask |= (ts >= a) & (ts < b)
    return mask


def _session_end_ts(ts: np.ndarray, exchange: str) -> np.ndarray:
    """End-of-continuous-session timestamp for each row (0 outside session)."""
    end = np.zeros(ts.shape, dtype=np.int64)
    for a, b in session_bounds(exchange):
        end[(ts >= a) & (ts < b)] = b
    return end


@dataclass(frozen=True)
class InstrumentMeta:
    """Static per-instrument configuration for the simulator."""

    exchange: str  # "sse" | "szse"
    settlement: str  # "T+0" | "T+1" (category map from etf_backtest_params.yaml)
    etf_category: str = "equity_etf"
    tick_cny: float = 0.001
    lot: int = 100
    max_order_units: int = 1_000_000
    price_limit_pct: float = 0.10


@dataclass
class DaySim:
    """One instrument-day on the aligned 3s grid, session-filtered."""

    ts_ms: np.ndarray
    bid1_px: np.ndarray
    ask1_px: np.ndarray
    mid_px: np.ndarray
    last_px: np.ndarray
    bid1_qty: np.ndarray
    ask1_qty: np.ndarray
    depth_bid5: np.ndarray
    depth_ask5: np.ndarray
    z: np.ndarray
    tradable: np.ndarray


@dataclass
class DayResult:
    """Result of simulating one instrument-day.

    ``cash_pnl`` is the per-row mark-to-market PnL EXCLUDING fees;
    ``fees_cny`` and ``slippage_cost_cny`` are reported separately.
    ``sellable_end_units`` is the closing inventory (next day's sellable
    pool); ``t1_blocked_units`` is the sell demand blocked by the T+1 lock.
    """

    position: np.ndarray
    trades_units: np.ndarray
    cash_pnl: np.ndarray
    fees_cny: float
    slippage_cost_cny: float
    sellable_end_units: float
    t1_blocked_units: float


@dataclass
class BacktestResult:
    """Aggregated multi-day, multi-instrument backtest result."""

    per_day: pl.DataFrame  # date, instrument, pnl_cny, fees_cny, n_trades, t1_blocked_units, ...
    equity_curve: pl.DataFrame  # date, equity_cny, drawdown_cny
    total_pnl_cny: float
    total_fees_cny: float
    sharpe_annualized: float
    max_drawdown_cny: float  # most negative drawdown (<= 0)
    turnover_units_per_day: float
    realized_round_trip_cost_bps: float
    n_days: int
    n_trades: int


def _validate_day_arrays(day: DaySim) -> int:
    n = np.asarray(day.ts_ms).shape[0]
    for name in (
        "bid1_px",
        "ask1_px",
        "mid_px",
        "last_px",
        "bid1_qty",
        "ask1_qty",
        "depth_bid5",
        "depth_ask5",
        "z",
        "tradable",
    ):
        arr = np.asarray(getattr(day, name))
        if arr.shape != (n,):
            raise ValueError(f"DaySim.{name} has shape {arr.shape}, expected ({n},)")
    return n


def simulate_day(
    day: DaySim,
    meta: InstrumentMeta,
    costs: CostModel,
    rule: PositionRule,
    *,
    sellable_start_units: float = 0.0,
    slippage_ticks: float = 1.0,
    use_depth_impact: bool = True,
) -> DayResult:
    """Simulate one instrument-day against target positions from ``rule``.

    T+1 rule: sells are bounded by the sellable pool carried from the
    previous day (``sellable_start_units``); same-day buys never become
    sellable for ``settlement == "T+1"``.  T+0 categories may sell same-day
    buys.  Buys are always allowed (they expand tomorrow's sellable pool).
    Fills cross the spread plus ``slippage_ticks``, rounded conservatively to
    the ¥0.001 tick grid (buys ceil, sells floor), with an optional
    depth-aware impact overlay.

    Rows where the rule's decision input (``z[i - signal_lag_rows]``) is
    absent or NaN never trade -- inventory is not liquidated before the
    signal is available -- except that the forced flatten on the last
    tradable row before an intraday untradable gap (lunch break, flagged
    interval) executes unconditionally as a risk-management override.  The
    day's final row is exempt from the forced flatten so closing inventory
    chains into the next day.  Lot-size rounding applies (odd lots are only
    sellable as a full liquidation, in one order).

    Rows without a usable mark (mid absent/one-sided AND last absent) are
    unmarkable: they can never trade, contribute zero row PnL, and the
    telescoping mark-to-market resumes at the next markable row.
    """
    n = _validate_day_arrays(day)
    if sellable_start_units < 0:
        raise ValueError("sellable_start_units must be >= 0")
    if abs(meta.tick_cny - TICK_CNY) > 1e-12:
        raise ValueError(
            f"InstrumentMeta.tick_cny={meta.tick_cny} but the execution model "
            f"is pinned to the fund tick {TICK_CNY}"
        )
    lot = int(meta.lot)
    max_order = int(meta.max_order_units)
    if lot <= 0 or max_order <= 0:
        raise ValueError("meta.lot and meta.max_order_units must be > 0")
    if meta.settlement not in ("T+0", "T+1"):
        raise ValueError(f"settlement must be 'T+0' or 'T+1', got {meta.settlement!r}")
    if rule.max_position_units > max_order:
        raise ValueError(
            f"PositionRule.max_position_units={rule.max_position_units} exceeds the "
            f"exchange max single order of {max_order} units; the target could never "
            "be filled in one order"
        )
    if rule.max_position_units % lot != 0:
        raise ValueError(
            f"PositionRule.max_position_units={rule.max_position_units} must be a "
            f"multiple of the lot size {lot}"
        )
    t0 = meta.settlement == "T+0"

    ts = np.asarray(day.ts_ms)
    bid1 = np.asarray(day.bid1_px, dtype=np.float64)
    ask1 = np.asarray(day.ask1_px, dtype=np.float64)
    mid = np.asarray(day.mid_px, dtype=np.float64)
    last = np.asarray(day.last_px, dtype=np.float64)
    bid_qty = np.asarray(day.bid1_qty, dtype=np.float64)
    ask_qty = np.asarray(day.ask1_qty, dtype=np.float64)
    z = np.asarray(day.z, dtype=np.float64)
    tradable = np.asarray(day.tradable, dtype=bool)

    positions = np.zeros(n, dtype=np.float64)
    trades = np.zeros(n, dtype=np.float64)
    cash_pnl = np.zeros(n, dtype=np.float64)
    if n == 0:
        return DayResult(
            positions,
            trades,
            cash_pnl,
            0.0,
            0.0,
            float(sellable_start_units),
            0.0,
        )

    targets = position_from_z(z, rule, tradable)
    max_units = float(rule.max_position_units)
    targets = np.clip(targets, 0.0, max_units)  # no spot shorting in v1
    lag = int(rule.signal_lag_rows)

    mark = np.where(mid > 0, mid, last)
    # Rows with no usable mark (one-sided book AND no last price) cannot be
    # revalued.  They are skipped in the PnL recurrence instead of being
    # allowed to poison it: 0.0 * NaN == NaN would contaminate the whole
    # instrument-day even when nothing trades (real data: early-session
    # one-sided snapshots of illiquid LOFs before their first trade).
    mark_ok = np.isfinite(mark) & (mark > 0)

    holdings = float(sellable_start_units)
    sellable = holdings  # units sellable right now
    fees_total = 0.0
    slip_total = 0.0
    blocked_total = 0.0

    prev_mark = None  # set at the first markable row
    prev_holdings = holdings
    # Target last attempted; suppresses re-attempting a blocked sell every row.
    last_target = holdings

    for i in range(n):
        m_i = float(mark[i])
        tgt = float(targets[i])
        # Never carry a position into an intraday untradable gap (lunch break,
        # flagged data interval): flatten on the last tradable row before it.
        # The day's final row is exempt -- closing inventory chains into the
        # next day as the T+1 sellable pool (底仓 model).
        force_flat = bool(tradable[i]) and i + 1 < n and not bool(tradable[i + 1])
        if force_flat:
            tgt = 0.0
        fill_i = 0.0

        # A decision at row i consumes z[i - lag]; rows whose decision input
        # is absent (pre-lag rows, warm-up, signal dropout) never trade.
        j_dec = i - lag
        decision_ok = j_dec >= 0 and bool(np.isfinite(z[j_dec]))
        can_trade = (
            bool(tradable[i])
            and (decision_ok or force_flat)
            and bool(mark_ok[i])
        )
        if can_trade and tgt != last_target:
            last_target = tgt
            delta = tgt - holdings
            if delta > _EPS_UNITS:
                # ------------------------------ buy -----------------------
                qty = math.floor(delta / lot) * lot
                room = math.floor((max_units - holdings) / lot) * lot
                qty = min(qty, max_order, room)
                if qty >= lot:
                    fill = cross_spread_fill(
                        "buy", float(bid1[i]), float(ask1[i]),
                        slippage_ticks=slippage_ticks,
                    )
                    if math.isfinite(fill):
                        if use_depth_impact:
                            best_cny = float(ask1[i]) * float(ask_qty[i])
                            bp = depth_impact_bps(qty * fill, best_cny)
                            fill = fill * (1.0 + bp / 1e4)
                        # Exchange quotes are already inside the ±10% band;
                        # fills derived from them need no limit clamp.
                        fill = _ceil_tick(fill)
                        fees_total += side_cost_cny(
                            costs, fill, qty, etf_category=meta.etf_category
                        )
                        slip_total += qty * (fill - m_i)
                        holdings += qty
                        if t0:
                            sellable += qty
                        trades[i] = qty
                        fill_i = fill
            elif delta < -_EPS_UNITS:
                # ------------------------------ sell ----------------------
                want = min(-delta, holdings)
                allowed = min(want, sellable)
                blocked_total += want - allowed  # T+1 lock shortfall
                if tgt <= _EPS_UNITS and allowed >= holdings - _EPS_UNITS:
                    qty = holdings  # full liquidation: odd lots go in one order
                else:
                    qty = math.floor(allowed / lot) * lot
                qty = min(qty, max_order)
                if qty > 0:
                    fill = cross_spread_fill(
                        "sell", float(bid1[i]), float(ask1[i]),
                        slippage_ticks=slippage_ticks,
                    )
                    if math.isfinite(fill):
                        if use_depth_impact:
                            best_cny = float(bid1[i]) * float(bid_qty[i])
                            bp = depth_impact_bps(qty * fill, best_cny)
                            fill = fill * (1.0 - bp / 1e4)
                        fill = _floor_tick(fill)
                        if fill < TICK_CNY:
                            fill = TICK_CNY
                        fees_total += side_cost_cny(
                            costs, fill, qty, etf_category=meta.etf_category
                        )
                        slip_total += qty * (m_i - fill)
                        holdings -= qty
                        sellable -= qty
                        trades[i] = -qty
                        fill_i = fill

        positions[i] = holdings
        if mark_ok[i]:
            # First markable row: no prior valuation exists, so carried
            # inventory is valued at its first mark (zero gain), matching
            # the old prev_mark = mark[0] initialization.
            pm = prev_mark if prev_mark is not None else m_i
            cash_pnl[i] = (
                holdings * m_i - prev_holdings * pm - trades[i] * fill_i
            )
            prev_holdings = holdings
            prev_mark = m_i
        else:
            # Unmarkable row: contributes nothing; the telescoping sum
            # resumes at the next markable row.  trades[i] == 0 here by
            # construction (can_trade requires a mark).
            cash_pnl[i] = 0.0

    return DayResult(
        positions,
        trades,
        cash_pnl,
        float(fees_total),
        float(slip_total),
        float(holdings),
        float(blocked_total),
    )


def _empty_result() -> BacktestResult:
    return BacktestResult(
        per_day=pl.DataFrame(schema=_PER_DAY_SCHEMA),
        equity_curve=pl.DataFrame(schema=_EQUITY_SCHEMA),
        total_pnl_cny=0.0,
        total_fees_cny=0.0,
        sharpe_annualized=0.0,
        max_drawdown_cny=0.0,
        turnover_units_per_day=0.0,
        realized_round_trip_cost_bps=0.0,
        n_days=0,
        n_trades=0,
    )


def run_backtest(
    panel: pl.DataFrame,
    factor: str,
    horizon_s: int,
    meta: dict[str, InstrumentMeta],
    costs: CostModel,
    rule: PositionRule,
    *,
    dates: Sequence[str] | None = None,
    initial_inventory_units: dict[str, float] | None = None,
    z_window_rows: int = 100,
) -> BacktestResult:
    """Run the full cost-aware backtest of one factor at one horizon.

    The panel must carry the interchange columns (date, exchange, instrument,
    ts_ms, flags, mid/last/bid1/ask1 prices + quantities, depth_bid5/ask5)
    plus the factor column.  Processing is grouped by (instrument, date);
    days are chained per instrument so the T+1 sellable pool and the
    overnight mark-to-market gap are carried across sessions.  Rows are
    filtered to the continuous session (SZSE closing auction excluded), rows
    with flags bit0/bit1 set or one-sided books are untradable, and no
    position may be held inside ``horizon_s`` of a session boundary.
    """
    required = [
        "date",
        "exchange",
        "instrument",
        "ts_ms",
        "flags",
        "mid_px",
        "last_px",
        "bid1_px",
        "ask1_px",
        "bid1_qty",
        "ask1_qty",
        "depth_bid5",
        "depth_ask5",
        factor,
    ]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"panel lacks required columns: {missing}")
    if horizon_s <= 0:
        raise ValueError(f"horizon_s must be > 0, got {horizon_s}")
    if z_window_rows < 2:
        raise ValueError(f"z_window_rows must be >= 2, got {z_window_rows}")

    df = panel.select(required)
    if dates is not None:
        df = df.filter(pl.col("date").is_in([str(d) for d in dates]))
    if df.height == 0:
        return _empty_result()
    df = df.sort(["instrument", "date", "ts_ms"])

    horizon_ms = int(horizon_s) * 1000
    inventory = {
        str(k): float(v) for k, v in (initial_inventory_units or {}).items()
    }

    per_day_rows: list[dict] = []
    for (inst_raw,), idf in df.group_by("instrument", maintain_order=True):
        inst = str(inst_raw)
        chain = inventory.get(inst, 0.0)
        prev_close_mark: float | None = None

        for (date_raw,), ddf in idf.group_by("date", maintain_order=True):
            date = str(date_raw)
            exch = str(ddf["exchange"][0])
            m = meta.get(inst)
            if m is None:
                m = InstrumentMeta(exchange=exch, settlement="T+1")

            ts_all = ddf["ts_ms"].cast(pl.Int64).to_numpy()
            smask = _session_mask(ts_all, exch)
            if not bool(smask.any()):
                continue

            def col(name: str) -> np.ndarray:
                return ddf[name].cast(pl.Float64).to_numpy()[smask]

            ts = ts_all[smask].astype(np.int64)
            flags = ddf["flags"].cast(pl.Int64).to_numpy()[smask]
            bid1 = col("bid1_px")
            ask1 = col("ask1_px")
            mid = col("mid_px")
            last = col("last_px")
            bid_qty = col("bid1_qty")
            ask_qty = col("ask1_qty")
            depth_bid5 = col("depth_bid5")
            depth_ask5 = col("depth_ask5")
            fvals = col(factor)

            two_sided = (bid1 > 0) & (ask1 > 0) & (bid_qty > 0) & (ask_qty > 0)
            flags_ok = (flags & _BAD_FLAG_MASK) == 0
            end_ts = _session_end_ts(ts, exch)
            tradable = flags_ok & two_sided & ((end_ts - ts) >= horizon_ms)

            z = causal_zscore(fvals, int(z_window_rows))
            day_sim = DaySim(
                ts_ms=ts,
                bid1_px=bid1,
                ask1_px=ask1,
                mid_px=mid,
                last_px=last,
                bid1_qty=bid_qty,
                ask1_qty=ask_qty,
                depth_bid5=depth_bid5,
                depth_ask5=depth_ask5,
                z=z,
                tradable=tradable,
            )
            res = simulate_day(day_sim, m, costs, rule, sellable_start_units=chain)

            mark = np.where(mid > 0, mid, last)
            mark_ok = np.isfinite(mark) & (mark > 0)
            gap = 0.0
            if prev_close_mark is not None and bool(mark_ok.any()):
                # First MARKABLE row: unmarkable rows (one-sided book, no
                # last) contribute no gap instead of poisoning it with NaN.
                gap = chain * (float(mark[mark_ok][0]) - float(prev_close_mark))

            net_pnl = float(res.cash_pnl.sum()) - res.fees_cny + gap
            traded_notional = float(
                (np.abs(res.trades_units) * np.where(mark_ok, mark, 0.0)).sum()
            )
            per_day_rows.append(
                {
                    "date": date,
                    "instrument": inst,
                    "pnl_cny": net_pnl,
                    "fees_cny": float(res.fees_cny),
                    "n_trades": int(np.count_nonzero(res.trades_units)),
                    "t1_blocked_units": float(res.t1_blocked_units),
                    "slippage_cost_cny": float(res.slippage_cost_cny),
                    "gap_pnl_cny": float(gap),
                    "position_end_units": float(res.sellable_end_units),
                    "traded_units": float(np.abs(res.trades_units).sum()),
                    "traded_notional_cny": traded_notional,
                }
            )
            chain = float(res.sellable_end_units)
            if mark.size > 0:
                prev_close_mark = float(mark[-1])

    if not per_day_rows:
        return _empty_result()

    per_day = pl.DataFrame(per_day_rows)
    total_pnl = float(per_day["pnl_cny"].sum())
    total_fees = float(per_day["fees_cny"].sum())

    daily = (
        per_day.group_by("date")
        .agg(pl.col("pnl_cny").sum().alias("pnl_cny"))
        .sort("date")
    )
    equity_curve = daily.with_columns(
        pl.col("pnl_cny").cum_sum().alias("equity_cny"),
    ).with_columns(
        (pl.col("equity_cny") - pl.col("equity_cny").cum_max()).alias("drawdown_cny")
    )

    n_days = int(per_day["date"].n_unique())
    pnl_arr = daily["pnl_cny"].to_numpy()
    sharpe = 0.0
    if pnl_arr.size >= 2:
        sd = float(pnl_arr.std(ddof=1))
        if sd > 0:
            sharpe = float(pnl_arr.mean() / sd * math.sqrt(ANN_TRADING_DAYS))

    turnover = float(per_day["traded_units"].sum()) / n_days if n_days else 0.0
    traded_notional_total = float(per_day["traded_notional_cny"].sum())
    total_slippage = float(per_day["slippage_cost_cny"].sum())
    rt_cost = 0.0
    if traded_notional_total > 0:
        # fees+slippage per side-traded notional, doubled to a round trip
        rt_cost = 2.0 * (total_fees + total_slippage) / traded_notional_total * 1e4

    return BacktestResult(
        per_day=per_day,
        equity_curve=equity_curve,
        total_pnl_cny=total_pnl,
        total_fees_cny=total_fees,
        sharpe_annualized=sharpe,
        max_drawdown_cny=float(equity_curve["drawdown_cny"].min()),
        turnover_units_per_day=turnover,
        realized_round_trip_cost_bps=rt_cost,
        n_days=n_days,
        n_trades=int(per_day["n_trades"].sum()),
    )
