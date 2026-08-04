"""Tests for backtest.derived: iter-001 admitted factors recomputed from
panel base columns (depth5_delta_60s, flow_divergence_300s).

The reference formulas mirror explore-specs-iter001/*.py; every expectation
here is computed independently (numpy) so the tests would catch any drift
between the backtest materialization and the admitted screen spec.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hft_autofactor.backtest.costs import CostModel
from hft_autofactor.backtest.derived import (
    DERIVED_FACTORS,
    is_derived,
    materialize_derived,
)
from hft_autofactor.backtest.engine import InstrumentMeta, run_backtest
from hft_autofactor.backtest.signals import PositionRule

ZERO_COSTS = CostModel(
    name="zero",
    commission_rate=0.0,
    min_commission_cny=0.0,
    handling_fee_rate=0.0,
    regulatory_fee_rate=0.0,
    transfer_fee_rate=0.0,
    stamp_duty_rate=0.0,
)


def _panel(
    n: int,
    *,
    date: str = "20250901",
    depth_bid: list[float] | None = None,
    depth_ask: list[float] | None = None,
    ofi: list[float] | None = None,
    ti: list[float] | None = None,
    ts_start_ms: int = 34_200_000,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date] * n,
            "exchange": ["sse"] * n,
            "instrument": ["588000"] * n,
            "ts_ms": [ts_start_ms + 3000 * i for i in range(n)],
            "snap_seq": list(range(n)),
            "flags": [0] * n,
            "mid_px": [1.300] * n,
            "last_px": [1.300] * n,
            "bid1_px": [1.299] * n,
            "ask1_px": [1.301] * n,
            "bid1_qty": [10_000.0] * n,
            "ask1_qty": [10_000.0] * n,
            "depth_bid5": depth_bid if depth_bid is not None else [1e6] * n,
            "depth_ask5": depth_ask if depth_ask is not None else [1e6] * n,
            "ofi_60s": ofi if ofi is not None else [0.0] * n,
            "trade_imbalance_60s": ti if ti is not None else [0.0] * n,
        }
    )


# --------------------------------------------------------------------------- #
# depth5_delta_60s
# --------------------------------------------------------------------------- #
def test_depth5_delta_warmup_and_values() -> None:
    n = 25
    bid = [100.0 + i for i in range(n)]
    ask = [100.0] * n
    out = materialize_derived(_panel(n, depth_bid=bid, depth_ask=ask), "depth5_delta_60s")
    vals = out["depth5_delta_60s"].to_numpy()

    imb = np.array([(b - a) / (b + a) for b, a in zip(bid, ask)])
    # first 20 rows: diff warm-up -> null (never zero-filled)
    assert np.isnan(vals[:20]).all()
    assert vals[20] == pytest.approx(imb[20] - imb[0])
    assert vals[24] == pytest.approx(imb[24] - imb[4])


def test_depth5_delta_zero_total_depth_is_null() -> None:
    n = 25
    bid = [100.0] * n
    ask = [100.0] * n
    bid[22] = 0.0  # imbalance at row 22 undefined (tot == 0)
    ask[22] = 0.0
    out = materialize_derived(_panel(n, depth_bid=bid, depth_ask=ask), "depth5_delta_60s")
    vals = out["depth5_delta_60s"].to_numpy()
    assert np.isnan(vals[22])  # imb[22] null -> diff null
    # row 20..21 fine (imb defined on both ends)
    assert np.isfinite(vals[20]) and np.isfinite(vals[21])


def test_depth5_delta_no_cross_day_leakage() -> None:
    n = 25
    bid1 = [100.0 + i for i in range(n)]
    bid2 = [300.0 - i for i in range(n)]
    ask = [100.0] * n
    d1 = _panel(n, date="20250901", depth_bid=bid1, depth_ask=ask)
    d2 = _panel(n, date="20250902", depth_bid=bid2, depth_ask=ask)
    out = materialize_derived(pl.concat([d1, d2]), "depth5_delta_60s")

    v2 = out.filter(pl.col("date") == "20250902")["depth5_delta_60s"].to_numpy()
    assert np.isnan(v2[:20]).all()  # warm-up restarts on day 2
    imb2 = np.array([(b - a) / (b + a) for b, a in zip(bid2, ask)])
    assert v2[20] == pytest.approx(imb2[20] - imb2[0])


# --------------------------------------------------------------------------- #
# flow_divergence_300s
# --------------------------------------------------------------------------- #
def test_flow_divergence_warmup_and_constant_neutral() -> None:
    n = 150
    out = materialize_derived(
        _panel(n, ofi=[5.0] * n, ti=[2.0] * n), "flow_divergence_300s"
    )
    vals = out["flow_divergence_300s"].to_numpy()
    assert np.isnan(vals[:99]).all()  # 100-row trailing window warm-up
    # constant windows => std 0 => z mapped to 0.0 => divergence 0.0
    assert np.isfinite(vals[99:]).all()
    assert np.abs(vals[99:]).max() == pytest.approx(0.0)


def test_flow_divergence_matches_manual_zscore() -> None:
    rng = np.random.default_rng(20260804)
    n = 220
    ofi = rng.normal(0.0, 1.0, n)
    ti = rng.normal(0.2, 0.8, n)
    out = materialize_derived(
        _panel(n, ofi=ofi.tolist(), ti=ti.tolist()), "flow_divergence_300s"
    )
    vals = out["flow_divergence_300s"].to_numpy()

    wins = np.lib.stride_tricks.sliding_window_view
    w = 100
    # spec: polars rolling_std default ddof=1
    m_ofi = wins(ofi, w).mean(axis=1)
    s_ofi = wins(ofi, w).std(axis=1, ddof=1)
    m_ti = wins(ti, w).mean(axis=1)
    s_ti = wins(ti, w).std(axis=1, ddof=1)
    z_ofi = (ofi[w - 1:] - m_ofi) / s_ofi
    z_ti = (ti[w - 1:] - m_ti) / s_ti
    expected = z_ofi - z_ti
    assert vals[99:] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert np.isnan(vals[:99]).all()


def test_flow_divergence_missing_source_raises() -> None:
    panel = _panel(30).drop("trade_imbalance_60s")
    with pytest.raises(ValueError, match="trade_imbalance_60s"):
        materialize_derived(panel, "flow_divergence_300s")


def test_materialize_unknown_factor_raises() -> None:
    with pytest.raises(KeyError):
        materialize_derived(_panel(5), "not_a_factor")


def test_is_derived_registry() -> None:
    assert is_derived("depth5_delta_60s")
    assert is_derived("flow_divergence_300s")
    assert not is_derived("oir")
    assert DERIVED_FACTORS["flow_divergence_300s"].sources == (
        "ofi_60s",
        "trade_imbalance_60s",
    )


def test_materialize_preserves_input_row_order() -> None:
    rng = np.random.default_rng(7)
    n = 60
    bid = [100.0 + i for i in range(n)]
    ask = [120.0 - 0.5 * i for i in range(n)]
    panel = _panel(n, depth_bid=bid, depth_ask=ask)
    shuffled = panel.sample(fraction=1.0, shuffle=True, seed=42)

    direct = materialize_derived(panel, "depth5_delta_60s")
    via_shuffle = materialize_derived(shuffled, "depth5_delta_60s").sort(
        ["date", "ts_ms"]
    )
    a = direct["depth5_delta_60s"].to_numpy()
    b = via_shuffle["depth5_delta_60s"].to_numpy()
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(a)
    assert a[mask] == pytest.approx(b[mask], rel=0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# end-to-end: run_backtest on a materialized derived column
# --------------------------------------------------------------------------- #
def test_run_backtest_with_derived_depth_factor() -> None:
    n = 40
    # depth imbalance ramps up strongly mid-day -> positive delta spike
    bid = [1e6] * n
    ask = [1e6] * n
    for i in range(25, 40):
        bid[i] = 5e6
    panel = _panel(n, depth_bid=bid, depth_ask=ask)
    panel = materialize_derived(panel, "depth5_delta_60s")

    rule = PositionRule(entry_z=2.0, exit_z=0.5, max_position_units=1000)
    meta = {"588000": InstrumentMeta(exchange="sse", settlement="T+0")}
    res = run_backtest(
        panel, "depth5_delta_60s", 15, meta, ZERO_COSTS, rule, z_window_rows=10
    )
    assert res.n_days == 1
    assert np.isfinite(res.total_pnl_cny)
    assert res.per_day["mark_end_cny"][0] == pytest.approx(1.300)
    assert res.per_day["peak_position_units"][0] >= 0.0
