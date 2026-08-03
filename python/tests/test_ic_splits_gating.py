"""Tests for IC math, Newey-West adjustment, splits, and gating gates."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from hft_autofactor.eval.gating import (
    GateConfig,
    TrialLedger,
    bhy_critical_values,
    deflated_sharpe_pvalue,
    norm_cdf,
    norm_ppf,
    stage1_screen,
    stage2_oos_gate,
    t_hurdle,
)
from hft_autofactor.eval.ic import (
    ICStats,
    ic_stats,
    newey_west_n_eff,
    rank_ic_cross_section,
    rank_ic_time_series,
    spearman,
)
from hft_autofactor.eval.splits import Split, is_oos_retention, purged_day_splits


# --------------------------------------------------------------------- #
# spearman                                                              #
# --------------------------------------------------------------------- #
def test_spearman_monotone_and_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, x * 3 + 7) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_known_value_with_ties():
    x = [1, 2, 3, 4, 5]
    y = [5, 6, 7, 8, 7]
    # ranks y = [1,2,3.5,5,3.5]; Pearson with 1..5 = 8/sqrt(95)
    assert spearman(np.array(x, float), np.array(y, float)) == pytest.approx(
        8.0 / math.sqrt(95.0)
    )


def test_spearman_nan_handling_and_degenerate():
    x = np.array([1.0, np.nan, 3.0, 4.0])
    y = np.array([2.0, 3.0, np.nan, 8.0])
    # pairwise complete: (1,2),(4,8) -> 2 obs, monotone -> 1.0
    assert spearman(x, y) == pytest.approx(1.0)
    assert math.isnan(spearman(np.array([1.0, np.nan]), np.array([2.0, 3.0])))
    assert math.isnan(spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))


# --------------------------------------------------------------------- #
# Newey-West n_eff                                                      #
# --------------------------------------------------------------------- #
def test_n_eff_bounds_and_persistence():
    rng = np.random.default_rng(7)
    white = rng.standard_normal(500)
    n_eff_w = newey_west_n_eff(white)
    assert 1.0 <= n_eff_w <= 500
    assert n_eff_w > 500 / 3  # near-iid keeps most of the information

    # random walk: strongly positive autocorrelation -> much smaller n_eff
    walk = np.cumsum(rng.standard_normal(500))
    n_eff_p = newey_west_n_eff(walk, max_lag=50)
    assert n_eff_p < n_eff_w
    assert n_eff_p < 500


def test_n_eff_small_and_constant():
    assert newey_west_n_eff(np.array([1.0])) == 1.0
    assert newey_west_n_eff(np.array([2.0, 2.0, 2.0, 2.0])) == 1.0
    assert newey_west_n_eff(np.array([])) == 0.0


# --------------------------------------------------------------------- #
# rank IC on a synthetic panel                                          #
# --------------------------------------------------------------------- #
def _toy_panel(n_inst=2, days=("20250602", "20250603"), n_rows=40, noise=0.05,
               seed=11):
    rng = np.random.default_rng(seed)
    records = []
    for d in days:
        for i in range(n_inst):
            inst = f"ETF{i}"
            for r in range(n_rows):
                f = rng.standard_normal()
                y = f + noise * rng.standard_normal()
                records.append(
                    {
                        "date": d,
                        "instrument": inst,
                        "ts_ms": 34_200_000 + r * 3000,
                        "oir": f,
                        "fwd_mid_ret_60s": y,
                    }
                )
    return pl.DataFrame(records)


def test_rank_ic_time_series_high_signal():
    panel = _toy_panel()
    ic_df = rank_ic_time_series(panel, "oir", 60)
    assert set(ic_df.columns) == {"date", "instrument", "ic", "n"}
    assert ic_df.height == 4  # 2 instruments x 2 days
    assert (ic_df["n"].to_numpy() == 40).all()
    assert (ic_df["ic"].to_numpy() > 0.9).all()


def test_rank_ic_cross_section_min_instruments():
    rng = np.random.default_rng(3)
    records = []
    for i in range(6):
        f = rng.standard_normal()
        records.append(
            {
                "date": "20250603",
                "instrument": f"ETF{i}",
                "ts_ms": 34_200_000,
                "oir": f,
                "fwd_mid_ret_60s": f + 0.05 * rng.standard_normal(),
            }
        )
    # a second timestamp with only 2 instruments -> dropped at min_instruments=5
    for i in range(2):
        records.append(
            {
                "date": "20250603",
                "instrument": f"ETF{i}",
                "ts_ms": 34_203_000,
                "oir": 0.1 * i,
                "fwd_mid_ret_60s": 0.1 * i,
            }
        )
    panel = pl.DataFrame(records)
    xs = rank_ic_cross_section(panel, "oir", 60, min_instruments=5)
    assert xs.height == 1
    assert xs["n_instruments"][0] == 6
    assert xs["ic"][0] > 0.8


def test_ic_stats_fields():
    panel = _toy_panel()
    ic_df = rank_ic_time_series(panel, "oir", 60)
    st = ic_stats(ic_df, "oir", 60, max_lag=20)
    assert isinstance(st, ICStats)
    assert st.factor == "oir" and st.horizon_s == 60
    assert st.n_obs == 4
    assert st.mean_ic > 0.9
    assert st.icir > 0
    assert st.win_rate == pytest.approx(1.0)
    assert 1.0 <= st.n_eff <= 4.0
    assert st.t_stat_nw > 0


def test_ic_stats_empty():
    empty = pl.DataFrame({"ic": []}, schema={"ic": pl.Float64})
    st = ic_stats(empty, "x", 15)
    assert st.n_obs == 0
    assert math.isnan(st.mean_ic)


# --------------------------------------------------------------------- #
# splits                                                                #
# --------------------------------------------------------------------- #
DATES = [f"202506{d:02d}" for d in range(1, 13)]


def test_purged_day_splits_anchored_embargo():
    splits = purged_day_splits(DATES, n_test_days=3, mode="anchored", embargo_days=1)
    assert len(splits) == 3
    s0 = splits[0]
    assert s0.test_dates == tuple(DATES[3:6])
    assert s0.train_dates == tuple(DATES[:2])  # DATES[2] embargoed
    # embargo day sits strictly between train end and test start
    for s in splits:
        assert DATES.index(s.train_dates[-1]) == DATES.index(s.test_dates[0]) - 2
    # anchored: train grows
    assert len(splits[0].train_dates) < len(splits[1].train_dates) < len(splits[2].train_dates)


def test_purged_day_splits_rolling():
    splits = purged_day_splits(
        DATES, n_test_days=3, mode="rolling", rolling_train_days=4, embargo_days=1
    )
    assert all(len(s.train_dates) <= 4 for s in splits)
    assert all(len(s.test_dates) == 3 for s in splits)


def test_purged_day_splits_validation_errors():
    with pytest.raises(ValueError):
        purged_day_splits(DATES, mode="bogus")
    with pytest.raises(ValueError):
        purged_day_splits(DATES, mode="rolling", rolling_train_days=None)
    with pytest.raises(ValueError):
        purged_day_splits(DATES, n_test_days=0)
    assert purged_day_splits(["20250601"], n_test_days=3) == []


def test_split_is_frozen():
    s = Split(train_dates=("a",), test_dates=("b",))
    with pytest.raises(Exception):
        s.train_dates = ("c",)  # type: ignore[misc]


@pytest.mark.parametrize(
    "is_ic,oos_ic,expected",
    [
        (0.04, 0.02, True),
        (0.04, 0.01, False),
        (0.04, -0.03, False),
        (0.0, 0.01, False),
        (-0.04, -0.025, True),
        (-0.04, 0.03, False),
        (0.04, float("nan"), False),
    ],
)
def test_is_oos_retention(is_ic, oos_ic, expected):
    assert is_oos_retention(is_ic, oos_ic, min_retention=0.5) is expected


# --------------------------------------------------------------------- #
# gating primitives                                                     #
# --------------------------------------------------------------------- #
def test_t_hurdle_scaling():
    assert t_hurdle(2) == pytest.approx(3.0)  # sqrt(2 ln 2) < 3 -> floor binds
    assert t_hurdle(100) == pytest.approx(math.sqrt(2 * math.log(100)))
    assert t_hurdle(10_000) > t_hurdle(100)


def test_bhy_critical_values_shape():
    crit = bhy_critical_values(10, q=0.10)
    c_m = sum(1.0 / i for i in range(1, 11))
    assert crit.shape == (10,)
    assert crit[-1] == pytest.approx(0.10 / c_m)
    assert (np.diff(crit) > 0).all()
    assert bhy_critical_values(0).size == 0


def test_norm_helpers():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-4)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)


def test_deflated_sharpe_monotone_and_bounded():
    p_strong = deflated_sharpe_pvalue(sr=2.0, n_trials=1, T=100, skew=0.0, kurt=3.0)
    p_weak = deflated_sharpe_pvalue(sr=0.1, n_trials=1, T=100, skew=0.0, kurt=3.0)
    assert 0.0 <= p_strong < p_weak <= 1.0
    assert p_strong < 0.05
    # more trials -> higher bar -> larger p-value for the same SR
    p_many = deflated_sharpe_pvalue(sr=0.5, n_trials=1000, T=250, skew=0.0, kurt=3.0)
    p_one = deflated_sharpe_pvalue(sr=0.5, n_trials=1, T=250, skew=0.0, kurt=3.0)
    assert p_many > p_one
    # degenerate inputs are safe
    assert deflated_sharpe_pvalue(sr=1.0, n_trials=5, T=2, skew=0.0, kurt=3.0) == 1.0


def test_trial_ledger(tmp_path):
    ledger = TrialLedger(tmp_path / "reports" / "trial_ledger.jsonl")
    assert ledger.n_trials() == 0
    ledger.log("oir", 60, {"window": 20}, "stage1", {"mean_ic": 0.03})
    ledger.log("wdi", 15, {}, "stage1", {"mean_ic": 0.01})
    ledger.log("oir", 60, {}, "stage2_walkforward", {"oos_mean_ic": 0.02})
    assert ledger.n_trials() == 3
    assert ledger.n_trials("stage1") == 2
    assert ledger.n_trials("stage2_walkforward") == 1


def test_stage1_screen_separates_strong_from_weak(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    strong = ICStats("oir", 60, n_obs=200, mean_ic=0.05, ic_std=0.04,
                     icir=1.25, t_stat_nw=4.5, n_eff=180.0, win_rate=0.7)
    weak = ICStats("wdi", 60, n_obs=200, mean_ic=0.001, ic_std=0.05,
                   icir=0.02, t_stat_nw=0.3, n_eff=150.0, win_rate=0.51)
    df = stage1_screen([strong, weak], ledger, GateConfig(),
                       noise_floors={("oir", 60): 0.005, ("wdi", 60): 0.005})
    assert df.height == 2
    by_factor = {r["factor"]: r for r in df.to_dicts()}
    assert by_factor["oir"]["passed"] is True
    assert by_factor["wdi"]["passed"] is False
    assert by_factor["oir"]["fdr_pass"] is True
    assert by_factor["oir"]["t_hurdle_min"] >= 3.0
    # ledger appended before thresholds were read
    assert ledger.n_trials("stage1") == 2


def test_stage1_screen_below_noise_floor_fails(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    # clears the level (0.025 >= 0.02), ICIR and t gates, but sits below the
    # permutation noise floor -> must fail on that gate alone
    st = ICStats("oir", 60, n_obs=200, mean_ic=0.025, ic_std=0.02,
                 icir=1.25, t_stat_nw=5.0, n_eff=180.0, win_rate=0.7)
    df = stage1_screen([st], ledger, GateConfig(), noise_floors={("oir", 60): 0.03})
    assert df["passed"][0] is False


def test_stage2_oos_gate_pass_and_fail():
    cfg = GateConfig()
    is_st = ICStats("oir", 60, 100, 0.04, 0.03, 1.33, 5.0, 90.0, 0.7)
    oos_ok = ICStats("oir", 60, 30, 0.025, 0.03, 0.83, 2.5, 25.0, 0.6)
    passed, details = stage2_oos_gate(is_st, oos_ok, cfg)
    assert passed is True
    assert details["retention_ok"] and details["sign_ok"]

    oos_decay = ICStats("oir", 60, 30, 0.01, 0.03, 0.33, 2.5, 25.0, 0.6)
    passed2, details2 = stage2_oos_gate(is_st, oos_decay, cfg)
    assert passed2 is False
    assert details2["retention_ok"] is False

    oos_flip = ICStats("oir", 60, 30, -0.03, 0.03, -1.0, 2.5, 25.0, 0.6)
    passed3, details3 = stage2_oos_gate(is_st, oos_flip, cfg)
    assert passed3 is False
    assert details3["sign_ok"] is False
