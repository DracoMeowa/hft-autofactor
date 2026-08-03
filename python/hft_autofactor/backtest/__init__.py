"""Cost-aware vectorized backtest engine for admitted A-share ETF factors.

Stage 5 of the hft-autofactor pipeline: maps a short-horizon factor to
positions (causal z-score + hysteresis rule with an actuation lag), simulates
execution against the L2 top-of-book with the fee_table_v1 cost stack, and
enforces A-share settlement rules (T+1 sell-lock for equity ETFs via
``sellable_qty(t) = holdings(t-1)`` inventory chaining; T+0 categories from
``etf_backtest_params.yaml``).  Reports net PnL, Sharpe, turnover, drawdown
and realized round-trip costs per commission scenario, and gates factor
admission on net-of-cost survival in ALL scenarios.
"""
from .costs import (
    HANDLING_FEE_EXEMPT_CATEGORIES,
    CostModel,
    load_cost_models,
    round_trip_cost_bps,
    side_cost_cny,
)
from .engine import (
    ANN_TRADING_DAYS,
    BacktestResult,
    DayResult,
    DaySim,
    InstrumentMeta,
    run_backtest,
    simulate_day,
)
from .execution import (
    TICK_CNY,
    clamp_price_limit,
    cross_spread_fill,
    depth_impact_bps,
)
from .metrics import gate_on_costs, summarize_results
from .signals import PositionRule, causal_zscore, position_from_z, zscore_column

__all__ = [
    # costs
    "HANDLING_FEE_EXEMPT_CATEGORIES",
    "CostModel",
    "load_cost_models",
    "side_cost_cny",
    "round_trip_cost_bps",
    # execution
    "TICK_CNY",
    "cross_spread_fill",
    "depth_impact_bps",
    "clamp_price_limit",
    # signals
    "causal_zscore",
    "zscore_column",
    "PositionRule",
    "position_from_z",
    # engine
    "ANN_TRADING_DAYS",
    "InstrumentMeta",
    "DaySim",
    "DayResult",
    "BacktestResult",
    "simulate_day",
    "run_backtest",
    # metrics
    "summarize_results",
    "gate_on_costs",
]
