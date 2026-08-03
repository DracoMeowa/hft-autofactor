"""Execution model: spread-crossing fills, depth impact, price-limit clamp.

All fills are taker fills rounded onto the fund tick grid (¥0.001):

* buys cross the ask and pay ``slippage_ticks`` extra ticks;
* sells hit the bid and give up ``slippage_ticks`` extra ticks.

On top of the flat slippage, a depth-aware impact overlay charges extra when
the order notional exceeds 10% of the displayed best-level depth (the liquid
tier of docs/knowledge/etf_backtest_params.yaml).  Impact grows linearly with
excess participation at rate ``kappa`` bp per unit of participation and is
capped at ``cap_bp``.  This keeps static spread assumptions from understating
the cost of size on mid/illiquid ETFs.

Fill prices are finally clamped against the ±10% price-limit band (rounded to
tick, SSE rule 3.3.17) when a prior close is supplied.
"""
from __future__ import annotations

import math

__all__ = [
    "TICK_CNY",
    "cross_spread_fill",
    "depth_impact_bps",
    "clamp_price_limit",
]

#: Minimum price increment for all SSE/SZSE funds (trading rules 3.3.11).
TICK_CNY: float = 0.001

#: Participations at/below this fraction of best-level depth are impact-free.
IMPACT_FREE_PARTICIPATION: float = 0.10


def round_to_tick(price: float) -> float:
    """Round ``price`` to the nearest ¥0.001 tick."""
    return round(round(price / TICK_CNY) * TICK_CNY, 6)


def _ceil_tick(price: float) -> float:
    return math.ceil(price / TICK_CNY - 1e-9) * TICK_CNY


def _floor_tick(price: float) -> float:
    return math.floor(price / TICK_CNY + 1e-9) * TICK_CNY


def cross_spread_fill(
    side: str,
    bid: float,
    ask: float,
    *,
    slippage_ticks: float = 1.0,
) -> float:
    """Taker fill price for crossing the spread, rounded to the tick grid.

    * ``buy``  -> ``ask + slippage_ticks * TICK_CNY`` (ceil to tick);
    * ``sell`` -> ``bid - slippage_ticks * TICK_CNY`` (floor to tick).

    Returns NaN when the quoted book is unusable (non-finite, non-positive,
    or crossed quotes).  ``slippage_ticks`` must be >= 0.
    """
    if slippage_ticks < 0:
        raise ValueError(f"slippage_ticks must be >= 0, got {slippage_ticks}")
    if not (math.isfinite(bid) and math.isfinite(ask)):
        return float("nan")
    if bid <= 0 or ask <= 0 or bid > ask:
        return float("nan")

    s = side.lower()
    if s == "buy":
        return round(_ceil_tick(ask + slippage_ticks * TICK_CNY), 6)
    if s == "sell":
        return round(_floor_tick(bid - slippage_ticks * TICK_CNY), 6)
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def depth_impact_bps(
    notional_cny: float,
    best_level_cny: float,
    *,
    kappa: float = 5.0,
    cap_bp: float = 50.0,
) -> float:
    """Extra impact in basis points for an order of ``notional_cny``.

    Impact is 0 while the order notional is <= 10% of the displayed
    best-level depth (``best_level_cny``).  Above that threshold it scales
    linearly with participation at ``kappa`` bp per unit participation and is
    capped at ``cap_bp``.  A non-positive best-level depth (hollow book)
    returns the cap directly.
    """
    if kappa < 0 or cap_bp < 0:
        raise ValueError("kappa and cap_bp must be >= 0")
    if notional_cny <= 0:
        return 0.0
    if not math.isfinite(notional_cny):
        raise ValueError("notional_cny must be finite")
    if best_level_cny <= 0 or not math.isfinite(best_level_cny):
        return cap_bp

    participation = notional_cny / best_level_cny
    if participation <= IMPACT_FREE_PARTICIPATION:
        return 0.0
    impact = kappa * (participation - IMPACT_FREE_PARTICIPATION)
    return min(impact, cap_bp)


def clamp_price_limit(
    price: float,
    pre_close: float,
    *,
    limit_pct: float = 0.10,
) -> float:
    """Clamp ``price`` into the ±``limit_pct`` band around ``pre_close``.

    Both band edges are rounded onto the tick grid (limit prices are tick
    multiples per SSE rule 3.3.17) and the result is tick-rounded as well.
    Non-positive ``pre_close`` disables clamping (only tick rounding applies).
    """
    if not math.isfinite(price):
        return float("nan")
    if pre_close > 0 and math.isfinite(pre_close) and limit_pct > 0:
        upper = _floor_tick(pre_close * (1.0 + limit_pct))
        lower = _ceil_tick(pre_close * (1.0 - limit_pct))
        price = min(max(price, lower), upper)
    return round_to_tick(price)
