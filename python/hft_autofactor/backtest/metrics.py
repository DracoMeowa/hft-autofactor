"""Cost-aware performance aggregation and the multi-scenario cost gate.

Factor admission requires net-of-cost survival in ALL commission scenarios
(institutional / retail_negotiated / retail_default) -- viability at 15-60s
horizons flips between them, so a factor that only survives the institutional
scenario is rejected.
"""
from __future__ import annotations

from typing import Mapping

import polars as pl

from .engine import ANN_TRADING_DAYS, BacktestResult

__all__ = ["summarize_results", "gate_on_costs"]

_SUMMARY_SCHEMA: dict[str, object] = {
    "scenario": pl.String,
    "n_days": pl.Int64,
    "n_trades": pl.Int64,
    "total_pnl_cny": pl.Float64,
    "gross_pnl_cny": pl.Float64,
    "total_fees_cny": pl.Float64,
    "total_slippage_cny": pl.Float64,
    "sharpe_annualized": pl.Float64,
    "ann_return_pct": pl.Float64,
    "hit_rate_days": pl.Float64,
    "max_drawdown_cny": pl.Float64,
    "max_drawdown_pct": pl.Float64,
    "capital_basis_cny": pl.Float64,
    "mean_daily_pnl_cny": pl.Float64,
    "avg_daily_traded_notional_cny": pl.Float64,
    "turnover_units_per_day": pl.Float64,
    "realized_round_trip_cost_bps": pl.Float64,
    "capacity_proxy_cny": pl.Float64,
}


def _capacity_proxy_cny(result: BacktestResult) -> float:
    """Capacity proxy: the largest single-day traded notional (CNY).

    The biggest day the strategy actually pushed through the book; reading
    PnL at larger notionals requires re-running with a bigger position and
    watching the depth-impact overlay and realized round-trip cost degrade.
    """
    if result.per_day.height == 0 or "traded_notional_cny" not in result.per_day.columns:
        return 0.0
    peak = result.per_day["traded_notional_cny"].max()
    return float(peak) if peak is not None else 0.0


def _capital_basis_cny(result: BacktestResult) -> float:
    """Capital basis (CNY) for return/drawdown ratios: peak position notional.

    Max over instrument-days of ``peak_position_units * mark_end_cny`` -- the
    largest capital the strategy actually had working at a closing mark.
    Returns 0.0 when the frame lacks the columns or never held a position.
    """
    per_day = result.per_day
    if (
        per_day.height == 0
        or "peak_position_units" not in per_day.columns
        or "mark_end_cny" not in per_day.columns
    ):
        return 0.0
    notionals = per_day["peak_position_units"] * per_day["mark_end_cny"]
    notionals = notionals.filter(notionals.is_finite())
    if notionals.len() == 0:
        return 0.0
    peak = notionals.max()
    return float(peak) if peak is not None else 0.0


def _daily_net_pnl(result: BacktestResult) -> pl.DataFrame:
    """Net PnL summed per calendar date across instruments."""
    if result.per_day.height == 0:
        return pl.DataFrame({"date": [], "pnl_cny": []})
    return (
        result.per_day.group_by("date")
        .agg(pl.col("pnl_cny").sum().alias("pnl_cny"))
    )


def _hit_rate_days(result: BacktestResult) -> float:
    """Fraction of trading days with positive NET PnL."""
    daily = _daily_net_pnl(result)
    if daily.height == 0:
        return 0.0
    wins = int((daily["pnl_cny"] > 0).sum())
    return wins / daily.height


def _sum_column(result: BacktestResult, name: str) -> float:
    if result.per_day.height == 0 or name not in result.per_day.columns:
        return 0.0
    total = result.per_day[name].sum()
    return float(total) if total is not None else 0.0


def summarize_results(results: Mapping[str, BacktestResult]) -> pl.DataFrame:
    """One summary row per commission scenario.

    PnL ladder: ``gross_pnl_cny`` is net PnL BEFORE the commission stack
    (spread/slippage/depth-impact are already inside the fills, hence inside
    gross); ``total_fees_cny`` is the commission drag; ``total_pnl_cny`` is
    fully net.  ``capital_basis_cny`` (peak position notional) converts the
    CNY figures into ``ann_return_pct`` / ``max_drawdown_pct``.
    """
    rows = []
    for name, r in results.items():
        capital = _capital_basis_cny(r)
        mean_daily = float(r.total_pnl_cny) / r.n_days if r.n_days > 0 else 0.0
        ann_return_pct = (
            mean_daily * ANN_TRADING_DAYS / capital * 100.0 if capital > 0 else 0.0
        )
        max_dd_pct = (
            float(r.max_drawdown_cny) / capital * 100.0 if capital > 0 else 0.0
        )
        rows.append(
            {
                "scenario": str(name),
                "n_days": int(r.n_days),
                "n_trades": int(r.n_trades),
                "total_pnl_cny": float(r.total_pnl_cny),
                "gross_pnl_cny": float(r.total_pnl_cny) + float(r.total_fees_cny),
                "total_fees_cny": float(r.total_fees_cny),
                "total_slippage_cny": _sum_column(r, "slippage_cost_cny"),
                "sharpe_annualized": float(r.sharpe_annualized),
                "ann_return_pct": float(ann_return_pct),
                "hit_rate_days": float(_hit_rate_days(r)),
                "max_drawdown_cny": float(r.max_drawdown_cny),
                "max_drawdown_pct": float(max_dd_pct),
                "capital_basis_cny": float(capital),
                "mean_daily_pnl_cny": float(mean_daily),
                "avg_daily_traded_notional_cny": (
                    _sum_column(r, "traded_notional_cny") / r.n_days
                    if r.n_days > 0
                    else 0.0
                ),
                "turnover_units_per_day": float(r.turnover_units_per_day),
                "realized_round_trip_cost_bps": float(r.realized_round_trip_cost_bps),
                "capacity_proxy_cny": _capacity_proxy_cny(r),
            }
        )
    if not rows:
        return pl.DataFrame(schema=_SUMMARY_SCHEMA)
    return pl.DataFrame(rows)


def gate_on_costs(
    results_by_scenario: Mapping[str, BacktestResult],
    *,
    min_net_sharpe: float = 0.5,
    min_days: int = 20,
) -> tuple[bool, dict]:
    """Gate a factor on net-of-cost survival in ALL commission scenarios.

    A scenario passes when the backtest covers at least ``min_days`` days,
    the annualized net Sharpe reaches ``min_net_sharpe``, and total net PnL
    is positive.  Returns ``(passed, details)`` where ``details`` carries the
    per-scenario verdicts and reasons.
    """
    if not results_by_scenario:
        return False, {
            "passed": False,
            "min_net_sharpe": min_net_sharpe,
            "min_days": min_days,
            "scenarios": {},
            "reason": "no commission scenarios were backtested",
        }

    details: dict[str, dict] = {}
    all_ok = True
    for name, r in results_by_scenario.items():
        reasons = []
        if r.n_days < min_days:
            reasons.append(f"only {r.n_days} days backtested (< {min_days})")
        if not (r.sharpe_annualized >= min_net_sharpe):
            reasons.append(
                f"net Sharpe {r.sharpe_annualized:.3f} < required {min_net_sharpe}"
            )
        if not (r.total_pnl_cny > 0):
            reasons.append(f"net PnL {r.total_pnl_cny:.2f} CNY <= 0")
        ok = not reasons
        all_ok = all_ok and ok
        details[str(name)] = {
            "pass": ok,
            "reasons": reasons,
            "n_days": int(r.n_days),
            "sharpe_annualized": float(r.sharpe_annualized),
            "total_pnl_cny": float(r.total_pnl_cny),
            "total_fees_cny": float(r.total_fees_cny),
            "max_drawdown_cny": float(r.max_drawdown_cny),
            "turnover_units_per_day": float(r.turnover_units_per_day),
            "realized_round_trip_cost_bps": float(r.realized_round_trip_cost_bps),
        }

    return all_ok, {
        "passed": all_ok,
        "min_net_sharpe": min_net_sharpe,
        "min_days": min_days,
        "scenarios": details,
    }
