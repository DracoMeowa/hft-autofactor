"""Cost model for A-share ETF backtests (fee_table_v1).

Per-side cost formula (see docs/knowledge/etf_cost_market_structure.md and the
machine-readable parameter file docs/knowledge/etf_backtest_params.yaml):

    side_cost = max(commission_rate * notional, min_commission_cny)
              + handling_fee_rate * notional     (0 for exempt categories:
                                                  money_etf / bond_etf)
              + regulatory_fee_rate * notional
              + transfer_fee_rate * notional
              + stamp_duty_rate * notional       (always 0 for ETF units:
                                                  Stamp Duty Law taxes stocks and
                                                  depository receipts only)

Key facts encoded here (fee_table_v1):

* Stamp duty is 0 for ETF units -- never set it non-zero for ETF backtests.
* Exchange handling fee (经手费) is 0.4bp per side for equity/commodity/
  cross-border ETFs and waived (暂免) for money and bond ETFs.
* Regulatory fee (证管费) and transfer fee (过户费) applicability to fund
  trades is unresolved; the base case carries 0 and the conservative
  sensitivity adds +0.2bp (regulatory) and +0.1bp (transfer) per side.
  Enable it with ``conservative_microfees=True`` until one real broker
  settlement statement settles both flags.
* Broker commission dominates. Three scenarios are loaded from the params
  file: ``institutional`` (0.5bp, min waived 免五), ``retail_negotiated``
  (1bp + ¥5/order minimum) and ``retail_default`` (2.5bp + ¥5/order minimum).
  Factor admission requires net-of-cost survival in ALL scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "HANDLING_FEE_EXEMPT_CATEGORIES",
    "CostModel",
    "load_cost_models",
    "side_cost_cny",
    "round_trip_cost_bps",
]

#: Categories exempt from the exchange handling fee (SSE 暂免; SZSE consistent).
#: Mirrors ``fees.handling_fee.exempt_categories`` in etf_backtest_params.yaml.
HANDLING_FEE_EXEMPT_CATEGORIES = frozenset({"money_etf", "bond_etf"})

#: Scenarios that must always be present in the params file.
REQUIRED_SCENARIOS = ("institutional", "retail_negotiated", "retail_default")

_BP = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Per-side fee stack for one commission scenario.

    All rates are fractions of notional, applied per side.
    ``min_commission_cny`` is the per-order minimum commission; 0.0 means the
    minimum is waived (免五).
    """

    name: str
    commission_rate: float  # per side, fraction of notional
    min_commission_cny: float  # per order; 0.0 = waived
    handling_fee_rate: float = 0.00004  # per side; 0 for money/bond ETF categories
    regulatory_fee_rate: float = 0.0
    transfer_fee_rate: float = 0.0
    stamp_duty_rate: float = 0.0  # always 0 for ETF units


def _read_yaml(params_yaml: str | Path) -> Mapping[str, Any]:
    path = Path(params_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"backtest params file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, Mapping):
        raise ValueError(f"backtest params file is not a mapping: {path}")
    return doc


def load_cost_models(
    params_yaml: str | Path, *, conservative_microfees: bool = False
) -> dict[str, CostModel]:
    """Load commission scenarios from ``etf_backtest_params.yaml``.

    Returns a dict keyed by scenario name; keys always include
    ``institutional``, ``retail_negotiated`` and ``retail_default``.

    With ``conservative_microfees=True`` the unresolved regulatory fee
    (+0.2bp/side) and transfer fee (+0.1bp/side) sensitivities are switched
    on; the base case carries 0 for both.
    """
    doc = _read_yaml(params_yaml)
    fees = doc.get("fees")
    if not isinstance(fees, Mapping):
        raise ValueError("backtest params file lacks a 'fees' section")

    # --- stamp duty: structurally 0 for ETF units -------------------------
    stamp = fees.get("stamp_duty") or {}
    stamp_rate = float(stamp.get("rate_per_side", 0.0))

    # --- handling fee + exemption list -------------------------------------
    handling = fees.get("handling_fee") or {}
    handling_rate = float(handling.get("rate_per_side", 0.00004))
    exempt_cfg = handling.get("exempt_categories")
    if exempt_cfg is not None and set(exempt_cfg) != set(HANDLING_FEE_EXEMPT_CATEGORIES):
        raise ValueError(
            "fees.handling_fee.exempt_categories in the params file "
            f"({sorted(exempt_cfg)}) diverges from the engine constant "
            f"({sorted(HANDLING_FEE_EXEMPT_CATEGORIES)}); update one of them"
        )

    # --- unresolved micro-fees: base 0 / conservative sensitivity ----------
    reg = fees.get("regulatory_fee") or {}
    reg_key = "rate_per_side_conservative" if conservative_microfees else "rate_per_side_base"
    regulatory_rate = float(reg.get(reg_key, 0.0))

    tra = fees.get("transfer_fee") or {}
    tra_key = "rate_per_side_conservative" if conservative_microfees else "rate_per_side_base"
    transfer_rate = float(tra.get(tra_key, 0.0))

    # --- commission scenarios ----------------------------------------------
    commission = fees.get("commission")
    if not isinstance(commission, Mapping):
        raise ValueError("fees.commission section missing in params file")
    scenarios = commission.get("scenarios")
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("fees.commission.scenarios missing in params file")

    models: dict[str, CostModel] = {}
    for name, sc in scenarios.items():
        if not isinstance(sc, Mapping):
            raise ValueError(f"commission scenario {name!r} is malformed")
        models[str(name)] = CostModel(
            name=str(name),
            commission_rate=float(sc["rate_per_side"]),
            min_commission_cny=float(sc.get("min_per_order_cny", 0.0)),
            handling_fee_rate=handling_rate,
            regulatory_fee_rate=regulatory_rate,
            transfer_fee_rate=transfer_rate,
            stamp_duty_rate=stamp_rate,
        )

    missing = [s for s in REQUIRED_SCENARIOS if s not in models]
    if missing:
        raise ValueError(f"commission scenarios missing from params file: {missing}")
    return models


def side_cost_cny(
    model: CostModel,
    price: float,
    qty: int,
    *,
    etf_category: str = "equity_etf",
) -> float:
    """Total regulatory + broker cost of ONE side (buy or sell) in CNY.

    ``price`` is the fill price in CNY, ``qty`` the number of fund units.
    The exchange handling fee is waived for money/bond ETF categories;
    stamp duty is never charged on ETF units.
    """
    if qty <= 0 or price <= 0:
        return 0.0
    notional = float(price) * float(qty)

    commission = model.commission_rate * notional
    if model.min_commission_cny > 0.0:
        commission = max(commission, model.min_commission_cny)

    if etf_category in HANDLING_FEE_EXEMPT_CATEGORIES:
        handling = 0.0
    else:
        handling = model.handling_fee_rate * notional

    regulatory = model.regulatory_fee_rate * notional
    transfer = model.transfer_fee_rate * notional
    stamp = model.stamp_duty_rate * notional  # 0 for ETF units by construction

    return commission + handling + regulatory + transfer + stamp


def round_trip_cost_bps(
    model: CostModel,
    price: float,
    lot_qty: int,
    *,
    etf_category: str = "equity_etf",
) -> float:
    """Fees-only round-trip cost in basis points of one-side notional.

    Buys ``lot_qty`` units at ``price`` and sells them back at the same price,
    returning ``(cost_buy + cost_sell) / notional * 1e4``.  Spread/slippage
    costs are NOT included here (they are handled by the execution model);
    this is the pure fee-stack component of the round-trip cost.
    """
    if lot_qty <= 0 or price <= 0:
        return 0.0
    notional = float(price) * float(lot_qty)
    total = 2.0 * side_cost_cny(model, price, lot_qty, etf_category=etf_category)
    return total / notional * 1e4
