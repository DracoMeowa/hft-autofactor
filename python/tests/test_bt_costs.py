"""Fee-stack tests against etf_backtest_params.yaml (fee_table_v1) scenarios.

Covers: scenario loading (institutional / retail_negotiated / retail_default),
the ¥5 minimum commission, handling-fee exemption for money/bond ETFs, zero
stamp duty, the conservative micro-fee sensitivity (+0.2bp regulatory +
0.1bp transfer per side) and round-trip cost magnitudes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hft_autofactor.backtest.costs import (
    HANDLING_FEE_EXEMPT_CATEGORIES,
    CostModel,
    load_cost_models,
    round_trip_cost_bps,
    side_cost_cny,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_YAML = """
fees:
  stamp_duty: {rate_per_side: 0.0}
  handling_fee:
    rate_per_side: 0.00004
    exempt_categories: [money_etf, bond_etf]
  regulatory_fee:
    rate_per_side_base: 0.0
    rate_per_side_conservative: 0.00002
  transfer_fee:
    rate_per_side_base: 0.0
    rate_per_side_conservative: 0.00001
  commission:
    scenarios:
      retail_default: {rate_per_side: 0.00025, min_per_order_cny: 5.0}
      retail_negotiated: {rate_per_side: 0.00010, min_per_order_cny: 5.0}
      institutional: {rate_per_side: 0.00005, min_per_order_cny: 0.0}
settlement:
  t_plus_1: [equity_etf]
  t_plus_0: [bond_etf, money_etf, gold_etf, commodity_etf,
             commodity_futures_etf, cross_border_etf]
"""


@pytest.fixture()
def synthetic_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "params.yaml"
    p.write_text(SYNTHETIC_YAML, encoding="utf-8")
    return p


def _real_yaml() -> Path | None:
    """Locate the real params file (pre- or post-move under docs/knowledge/)."""
    for rel in (
        "docs/knowledge/etf_backtest_params.yaml",
        "docs/etf_backtest_params.yaml",
    ):
        p = REPO_ROOT / rel
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- #
# Scenario loading
# --------------------------------------------------------------------------- #
def test_load_scenarios_and_rates(synthetic_yaml: Path) -> None:
    models = load_cost_models(synthetic_yaml)
    assert set(models) == {"institutional", "retail_negotiated", "retail_default"}

    inst = models["institutional"]
    assert inst.commission_rate == pytest.approx(0.00005)
    assert inst.min_commission_cny == 0.0  # 免五 waived
    assert inst.handling_fee_rate == pytest.approx(0.00004)
    assert inst.regulatory_fee_rate == 0.0
    assert inst.transfer_fee_rate == 0.0
    assert inst.stamp_duty_rate == 0.0

    neg = models["retail_negotiated"]
    assert neg.commission_rate == pytest.approx(0.00010)
    assert neg.min_commission_cny == pytest.approx(5.0)

    retail = models["retail_default"]
    assert retail.commission_rate == pytest.approx(0.00025)
    assert retail.min_commission_cny == pytest.approx(5.0)


def test_conservative_microfees_sensitivity(synthetic_yaml: Path) -> None:
    base = load_cost_models(synthetic_yaml)["institutional"]
    cons = load_cost_models(synthetic_yaml, conservative_microfees=True)["institutional"]
    assert base.regulatory_fee_rate == 0.0
    assert base.transfer_fee_rate == 0.0
    assert cons.regulatory_fee_rate == pytest.approx(0.00002)
    assert cons.transfer_fee_rate == pytest.approx(0.00001)
    # commission / handling untouched by the sensitivity switch
    assert cons.commission_rate == base.commission_rate
    assert cons.handling_fee_rate == base.handling_fee_rate


def test_real_params_yaml_matches_fee_table_v1() -> None:
    path = _real_yaml()
    if path is None:
        pytest.skip("real etf_backtest_params.yaml not present in repo")
    models = load_cost_models(path)
    assert {"institutional", "retail_negotiated", "retail_default"} <= set(models)
    assert models["institutional"].commission_rate == pytest.approx(0.00005)
    assert models["institutional"].min_commission_cny == 0.0
    assert models["retail_negotiated"].commission_rate == pytest.approx(0.00010)
    assert models["retail_negotiated"].min_commission_cny == pytest.approx(5.0)
    assert models["retail_default"].commission_rate == pytest.approx(0.00025)
    assert models["retail_default"].min_commission_cny == pytest.approx(5.0)
    for m in models.values():
        assert m.stamp_duty_rate == 0.0  # ETF units are never stamp-taxable


# --------------------------------------------------------------------------- #
# Side costs
# --------------------------------------------------------------------------- #
def test_min_commission_dominates_small_orders(synthetic_yaml: Path) -> None:
    retail = load_cost_models(synthetic_yaml)["retail_default"]
    # notional 1000 CNY: 2.5bp = 0.25 CNY commission -> the 5 CNY floor binds
    cost = side_cost_cny(retail, 1.0, 1000)
    assert cost == pytest.approx(5.0 + 0.00004 * 1000.0)


def test_institutional_side_cost_no_min(synthetic_yaml: Path) -> None:
    inst = load_cost_models(synthetic_yaml)["institutional"]
    # notional 40000 CNY: (0.5bp + 0.4bp) per side = 0.9bp = 3.6 CNY
    cost = side_cost_cny(inst, 4.0, 10_000)
    assert cost == pytest.approx(40_000 * (0.00005 + 0.00004))


def test_handling_fee_exemption_money_and_bond(synthetic_yaml: Path) -> None:
    inst = load_cost_models(synthetic_yaml)["institutional"]
    price, qty = 4.0, 100_000
    notional = price * qty
    equity = side_cost_cny(inst, price, qty, etf_category="equity_etf")
    assert HANDLING_FEE_EXEMPT_CATEGORIES == frozenset({"money_etf", "bond_etf"})
    for cat in ("money_etf", "bond_etf"):
        exempt = side_cost_cny(inst, price, qty, etf_category=cat)
        assert equity - exempt == pytest.approx(0.00004 * notional)


def test_zero_qty_or_price_costs_nothing(synthetic_yaml: Path) -> None:
    retail = load_cost_models(synthetic_yaml)["retail_default"]
    assert side_cost_cny(retail, 4.0, 0) == 0.0
    assert side_cost_cny(retail, 0.0, 1000) == 0.0


def test_round_trip_bps_institutional_fee_only(synthetic_yaml: Path) -> None:
    inst = load_cost_models(synthetic_yaml)["institutional"]
    # fees-only round trip = 2 * (0.5bp + 0.4bp) = 1.8bp for size without min
    rt = round_trip_cost_bps(inst, 4.0, 100_000)
    assert rt == pytest.approx(1.8)


def test_round_trip_bps_retail_small_size_inflated(synthetic_yaml: Path) -> None:
    retail = load_cost_models(synthetic_yaml)["retail_default"]
    # tiny notional: the ¥5 minimum blows the round-trip cost far past fees
    rt = round_trip_cost_bps(retail, 1.0, 100)
    assert rt > 500.0  # > 5% round trip: retail small size is untradeable


def test_conservative_side_cost_adds_0p3bp(synthetic_yaml: Path) -> None:
    base = load_cost_models(synthetic_yaml)["institutional"]
    cons = load_cost_models(synthetic_yaml, conservative_microfees=True)["institutional"]
    price, qty = 4.0, 100_000
    diff = side_cost_cny(cons, price, qty) - side_cost_cny(base, price, qty)
    assert diff == pytest.approx(0.00003 * price * qty)


# --------------------------------------------------------------------------- #
# Loader validation
# --------------------------------------------------------------------------- #
def test_missing_commission_section_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("fees:\n  stamp_duty: {rate_per_side: 0.0}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_cost_models(p)


def test_missing_scenario_raises(tmp_path: Path) -> None:
    p = tmp_path / "partial.yaml"
    p.write_text(
        """
fees:
  commission:
    scenarios:
      institutional: {rate_per_side: 0.00005, min_per_order_cny: 0.0}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_cost_models(p)


def test_divergent_exemption_list_raises(tmp_path: Path) -> None:
    p = tmp_path / "divergent.yaml"
    p.write_text(
        SYNTHETIC_YAML.replace(
            "exempt_categories: [money_etf, bond_etf]",
            "exempt_categories: [money_etf]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_cost_models(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cost_models(tmp_path / "nope.yaml")
