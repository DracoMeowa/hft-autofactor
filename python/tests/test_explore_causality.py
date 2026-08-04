"""Explore-lane runner + panel prefix causality tests.

Includes the REQUIRED leaky-prototype canary: a prototype computing a
forward shift must be caught by the truncate-and-recompute prefix test,
mirroring the canary semantics of validation/mask_test.py.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from conftest import make_day_rows, write_interchange_csv

from hft_autofactor.explore.causality import (
    choose_panel_cutoffs,
    compare_panel_prefix,
    panel_prefix_check,
)
from hft_autofactor.explore.registry import PrototypeRegistry, explore_prototype
from hft_autofactor.explore.runner import (
    PrototypeComputeError,
    compute_prototype_column,
    load_explore_panel,
    run_prototype,
)
from hft_autofactor.ingest import build_day_parquet

DATE_A = "20250602"
DATE_B = "20250603"
START_MS = 34_200_000  # 09:30:00
N_SNAP = 50


# --------------------------------------------------------------------- #
# synthetic panel builder (base + library factors + labels)             #
# --------------------------------------------------------------------- #
def make_panel(n_snap: int = N_SNAP, seed: int = 7) -> pl.DataFrame:
    """Two days x two instruments random-walk panel on the 3s grid."""
    rng = np.random.default_rng(seed)
    records = []
    for date in (DATE_A, DATE_B):
        for inst in ("510300", "510050"):
            mid = 4.0
            for i in range(n_snap):
                mid += rng.normal(0.0, 0.0005)
                ts = START_MS + i * 3000
                records.append(
                    {
                        "date": date,
                        "exchange": "sse",
                        "instrument": inst,
                        "ts_ms": ts,
                        "snap_seq": 1000 + i,
                        "flags": 0,
                        "mid_px": mid,
                        "last_px": mid + 0.0001,
                        "bid1_px": mid - 0.001,
                        "ask1_px": mid + 0.001,
                        "bid1_qty": 10000,
                        "ask1_qty": 8000,
                        "depth_bid5": 50000,
                        "depth_ask5": 42000,
                        "oir": rng.normal(0.0, 0.3),
                        "wdi": rng.normal(0.0, 0.3),
                        "fwd_mid_ret_60s": rng.normal(0.0, 1e-4),
                        "fwd_mid_ret_300s": rng.normal(0.0, 1e-4),
                    }
                )
    return pl.DataFrame(records)


def causal_proto(**overrides) -> "object":
    """Trailing 5-row mean of log-mid returns: strictly backward-looking."""
    kwargs = dict(
        name="causal_test_f",
        mechanism="test mechanism (backward rolling mean of 3s log-mid returns)",
        info_set="mid_px",
        inspiration="test fixture",
        compute=lambda part: part.select(
            pl.col("mid_px").log().diff(1).rolling_mean(window_size=5, min_samples=5)
        ).to_series(),
    )
    kwargs.update(overrides)
    return explore_prototype(**kwargs)


def leaky_shift_proto() -> "object":
    """DELIBERATELY LEAKY: value at t reads mid at t+5 rows (forward shift)."""
    return explore_prototype(
        name="leaky_shift_f",
        mechanism="canary: forward shift must be caught",
        info_set="mid_px",
        inspiration="mask-test canary analogue",
        compute=lambda part: part["mid_px"].shift(-5),
    )


def leaky_global_mean_proto() -> "object":
    """DELIBERATELY LEAKY: centers by the FULL-series mean (uses future)."""
    return explore_prototype(
        name="leaky_mean_f",
        mechanism="canary: whole-series moment must be caught",
        info_set="mid_px",
        inspiration="mask-test canary analogue",
        compute=lambda part: part["mid_px"] - part["mid_px"].mean(),
    )


def leaky_rank_proto() -> "object":
    """DELIBERATELY LEAKY: whole-series rank depends on every future row."""
    return explore_prototype(
        name="leaky_rank_f",
        mechanism="canary: whole-series rank must be caught",
        info_set="mid_px",
        inspiration="mask-test canary analogue",
        compute=lambda part: part["mid_px"].rank(),
    )


def leaky_reverse_cumsum_proto() -> "object":
    """DELIBERATELY LEAKY: cumsum from the END reads future rows."""
    return explore_prototype(
        name="leaky_rev_cumsum_f",
        mechanism="canary: reverse cumsum must be caught",
        info_set="mid_px",
        inspiration="mask-test canary analogue",
        compute=lambda part: part["mid_px"].reverse().cum_sum().reverse(),
    )


# --------------------------------------------------------------------- #
# cutoff selection                                                      #
# --------------------------------------------------------------------- #
def test_choose_panel_cutoffs_labels_and_order():
    panel = make_panel()
    pts = choose_panel_cutoffs(panel, k=4)
    assert [p.label for p in pts] == ["warmup", "mid_am", "post_lunch", "late"]
    keys = [(p.date, p.ts_ms) for p in pts]
    assert keys == sorted(keys)
    assert keys[0][0] == DATE_A and keys[-1][0] == DATE_B


def test_choose_panel_cutoffs_deterministic_and_fuzz():
    panel = make_panel()
    a = choose_panel_cutoffs(panel, k=7, seed=42)
    b = choose_panel_cutoffs(panel, k=7, seed=42)
    assert a == b
    assert sum(1 for p in a if p.label.startswith("fuzz_")) == 3
    one = choose_panel_cutoffs(panel, k=1)
    assert [p.label for p in one] == ["late"]
    with pytest.raises(ValueError):
        choose_panel_cutoffs(panel, k=0)


def test_choose_panel_cutoffs_empty_panel_raises():
    empty = make_panel().clear()
    with pytest.raises(ValueError):
        choose_panel_cutoffs(empty, k=4)


# --------------------------------------------------------------------- #
# the panel prefix causality test                                       #
# --------------------------------------------------------------------- #
def test_causal_prototype_passes_prefix_check():
    panel = make_panel()
    proto = causal_proto()
    report = panel_prefix_check(panel, proto, k=4)
    assert report.passed is True
    assert len(report.points) == 4 and len(report.diffs) == 4
    assert all(d.identical for d in report.diffs)
    assert all(d.n_rows_scope > 0 for d in report.diffs)
    assert report.prototype == proto.name


def test_leaky_forward_shift_prototype_is_rejected():
    """REQUIRED canary: a forward shift must fail the prefix identity test."""
    panel = make_panel()
    report = panel_prefix_check(panel, leaky_shift_proto(), k=4)
    assert report.passed is False
    bad = [d for d in report.diffs if not d.identical]
    assert bad, "at least one cutoff must expose the forward shift"
    assert "value mismatch" in bad[0].first_diff
    assert "leaky_shift_f" in bad[0].first_diff


def test_leaky_global_mean_prototype_is_rejected():
    panel = make_panel()
    report = panel_prefix_check(panel, leaky_global_mean_proto(), k=4)
    assert report.passed is False
    assert any(not d.identical for d in report.diffs)


def test_leaky_whole_series_rank_prototype_is_rejected():
    panel = make_panel()
    report = panel_prefix_check(panel, leaky_rank_proto(), k=4)
    assert report.passed is False
    assert any(not d.identical for d in report.diffs)


def test_leaky_reverse_cumsum_prototype_is_rejected():
    panel = make_panel()
    report = panel_prefix_check(panel, leaky_reverse_cumsum_proto(), k=4)
    assert report.passed is False
    assert any(not d.identical for d in report.diffs)


def test_backward_shift_control_is_causal():
    """A pure BACKWARD shift (lag) is causal and must pass the gate."""
    panel = make_panel()
    lagged = explore_prototype(
        name="lag_control_f",
        mechanism="control: backward shift is causal",
        info_set="mid_px",
        inspiration="mask-test canary control",
        compute=lambda part: part["mid_px"].shift(5),
    )
    report = panel_prefix_check(panel, lagged, k=4)
    assert report.passed is True


def test_prefix_check_more_cutoffs_still_catch_leak():
    panel = make_panel()
    report = panel_prefix_check(panel, leaky_shift_proto(), k=8, seed=11)
    assert len(report.points) == 8
    assert report.passed is False


def test_compare_prefix_detects_tampered_value():
    panel = make_panel()
    proto = causal_proto()
    aug = compute_prototype_column(panel, proto)
    cutoff = choose_panel_cutoffs(panel, k=1)[0]
    # tamper one in-scope cell of the full column
    scope = aug.filter(cutoff.scope_expr())
    key = scope.row(5, named=True)
    tampered = aug.with_columns(
        pl.when(
            (pl.col("date") == key["date"])
            & (pl.col("instrument") == key["instrument"])
            & (pl.col("ts_ms") == key["ts_ms"])
        )
        .then(pl.lit(999.0))
        .otherwise(pl.col(proto.name))
        .alias(proto.name)
    )
    trunc = compute_prototype_column(panel.filter(cutoff.scope_expr()), proto)
    diff = compare_panel_prefix(tampered, trunc, proto.name, cutoff)
    assert diff.identical is False
    assert "value mismatch" in diff.first_diff


def test_compare_prefix_null_equals_null_warmup():
    """Warm-up nulls on both sides are identical (not a mismatch)."""
    panel = make_panel()
    proto = causal_proto()
    aug = compute_prototype_column(panel, proto)
    cutoff = choose_panel_cutoffs(panel, k=1)[0]
    trunc = compute_prototype_column(panel.filter(cutoff.scope_expr()), proto)
    diff = compare_panel_prefix(aug, trunc, proto.name, cutoff)
    assert diff.identical is True
    assert aug[proto.name].null_count() > 0  # warm-up really present


def test_prefix_check_scales_linearly_on_large_panel():
    """Regression: the prefix comparison must stay ~linear in prefix rows.

    A previous revision of compare_panel_prefix rebuilt ``set(f_keys)``
    inside the membership comprehension (O(n^2)); on a full-day smoke panel
    (~10^5 rows in the late prefix) that hung for hours at 100% CPU.  This
    24k-row panel finishes in ~1s once linear; the quadratic version needs
    minutes, so the 30s budget is a wide, machine-independent margin.
    """
    import time

    panel = make_panel(n_snap=6000)  # 2 days x 2 instruments x 6000 rows
    proto = causal_proto()
    t0 = time.monotonic()
    report = panel_prefix_check(panel, proto, k=4)
    elapsed = time.monotonic() - t0
    assert report.passed is True
    assert elapsed < 30.0, f"prefix check took {elapsed:.1f}s (quadratic regression?)"


# --------------------------------------------------------------------- #
# compute_prototype_column contract                                     #
# --------------------------------------------------------------------- #
def test_compute_preserves_input_row_order_and_alignment():
    panel = make_panel().sort(["ts_ms", "instrument", "date"])  # scrambled
    proto = causal_proto()
    aug = compute_prototype_column(panel, proto)
    assert aug.height == panel.height
    assert aug.columns == panel.columns + [proto.name]
    # row-wise alignment: recompute on sorted panel, join back, compare
    expected = compute_prototype_column(
        panel.sort(["date", "instrument", "ts_ms"]), proto
    )
    joined = aug.join(
        expected.select(["date", "instrument", "ts_ms", proto.name]),
        on=["date", "instrument", "ts_ms"],
        suffix="_ref",
    )
    diffs = joined.filter(
        ~(
            (pl.col(proto.name) == pl.col(f"{proto.name}_ref"))
            | (pl.col(proto.name).is_null() & pl.col(f"{proto.name}_ref").is_null())
        )
    )
    assert diffs.is_empty()


def test_compute_strips_label_columns_from_info_set():
    """The compute spec receives NO label columns (targets invisible)."""
    panel = make_panel()
    seen: dict[str, list[str]] = {}

    def spy(part: pl.DataFrame) -> pl.Series:
        seen["cols"] = part.columns
        return part["mid_px"]

    proto = explore_prototype(
        name="spy_f", mechanism="m", info_set="mid_px",
        inspiration="i", compute=spy,
    )
    compute_prototype_column(panel, proto)
    assert "mid_px" in seen["cols"]
    assert not any(c.startswith("fwd_") for c in seen["cols"])


def test_compute_reading_a_label_column_cannot_succeed():
    """A prototype reaching for a label gets a missing-column error."""
    panel = make_panel()
    proto = explore_prototype(
        name="cheat_f", mechanism="m", info_set="fwd_mid_ret_60s",
        inspiration="i", compute=lambda part: part["fwd_mid_ret_60s"],
    )
    with pytest.raises(Exception) as excinfo:
        compute_prototype_column(panel, proto)
    assert "fwd_mid_ret_60s" in str(excinfo.value)


def test_compute_rejects_misaligned_return():
    panel = make_panel()
    proto = explore_prototype(
        name="short_f", mechanism="m", info_set="mid_px", inspiration="i",
        compute=lambda part: part["mid_px"].head(part.height - 1),
    )
    with pytest.raises(PrototypeComputeError, match="align"):
        compute_prototype_column(panel, proto)


def test_compute_rejects_bad_return_types():
    panel = make_panel()
    wide = explore_prototype(
        name="wide_f", mechanism="m", info_set="mid_px", inspiration="i",
        compute=lambda part: part.select(["mid_px", "last_px"]),
    )
    with pytest.raises(PrototypeComputeError, match="one column"):
        compute_prototype_column(panel, wide)
    scalar = explore_prototype(
        name="scalar_f", mechanism="m", info_set="mid_px", inspiration="i",
        compute=lambda part: 3.14,
    )
    with pytest.raises(PrototypeComputeError, match="returned"):
        compute_prototype_column(panel, scalar)


def test_compute_refuses_to_shadow_existing_column():
    panel = make_panel()
    # registry name rules reject 'oir' at registration already; construct
    # directly to prove the runner guard stands on its own
    from hft_autofactor.explore.registry import Prototype

    shadow = Prototype(
        name="oir", mechanism="m", info_set="mid_px",
        inspiration="i", compute=lambda part: part["mid_px"],
    )
    with pytest.raises(PrototypeComputeError, match="refusing to shadow"):
        compute_prototype_column(panel, shadow)


def test_compute_empty_panel_returns_empty_column():
    panel = make_panel().clear()
    proto = causal_proto()
    aug = compute_prototype_column(panel, proto)
    assert proto.name in aug.columns
    assert aug.height == 0
    assert aug[proto.name].dtype == pl.Float64


# --------------------------------------------------------------------- #
# runner over day partitions (conftest interchange CSV -> parquet)      #
# --------------------------------------------------------------------- #
FACTORS = ("oir", "wdi")


def _write_raw_day(cfg, date: str, instruments=("510300", "510050"), n_snap=N_SNAP):
    rows = []
    for inst in instruments:
        rows.extend(
            make_day_rows(inst, n_snap=n_snap, start_ms=START_MS, factors=FACTORS)
        )
    write_interchange_csv(
        cfg.raw_csv(date, "sse", 1), date=date, exchange="sse", rows=rows,
        factors=FACTORS,
    )


@pytest.fixture
def panel_cfg(small_cfg):
    """small_cfg with two days of parquet partitions ready to load."""
    for date in (DATE_A, DATE_B):
        _write_raw_day(small_cfg, date)
        build_day_parquet(date, small_cfg)
    return small_cfg


def test_run_prototype_ok_writes_partitions(panel_cfg):
    proto = causal_proto()
    result = run_prototype(panel_cfg, proto, [DATE_A, DATE_B], chunk_days=1, k=3)
    assert result.status == "ok"
    assert len(result.partitions) == 2
    assert all(p.is_file() for p in result.partitions)
    assert result.causality is not None and result.causality.passed
    assert result.report_path is not None and result.report_path.is_file()

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["causality"]["passed"] is True
    assert payload["prototype"]["mechanism"] == proto.mechanism

    panel = load_explore_panel(panel_cfg, proto.name, [DATE_A, DATE_B])
    assert proto.name in panel.columns
    assert "fwd_mid_ret_60s" in panel.columns  # labels kept for alignment
    assert panel.height == 2 * 2 * N_SNAP


def test_run_prototype_rejects_leaky_and_cleans_up(panel_cfg):
    proto = leaky_shift_proto()
    result = run_prototype(panel_cfg, proto, [DATE_A, DATE_B], chunk_days=1, k=3)
    assert result.status == "rejected_causality"
    assert result.partitions == []
    assert "prefix identity failed" in result.message
    for date in (DATE_A, DATE_B):
        from hft_autofactor.explore.layout import panel_path

        assert not panel_path(panel_cfg, proto.name, date).exists()
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected_causality"
    assert payload["causality"]["passed"] is False


def test_run_prototype_skip_if_done_and_overwrite(panel_cfg):
    proto = causal_proto()
    first = run_prototype(panel_cfg, proto, [DATE_A, DATE_B], chunk_days=2)
    assert first.status == "ok"
    mtime = first.partitions[0].stat().st_mtime_ns

    again = run_prototype(panel_cfg, proto, [DATE_A, DATE_B], chunk_days=2)
    assert again.status == "skipped"
    assert again.partitions[0].stat().st_mtime_ns == mtime

    forced = run_prototype(
        panel_cfg, proto, [DATE_A, DATE_B], chunk_days=2, overwrite=True
    )
    assert forced.status == "ok"
    assert forced.partitions[0].stat().st_mtime_ns != mtime


def test_run_prototype_chunking_is_bit_identical(panel_cfg):
    """chunk_days must not change values: (date, instrument) groups never split."""
    proto = causal_proto()
    run_prototype(panel_cfg, proto, [DATE_A, DATE_B], chunk_days=1)
    p1 = load_explore_panel(panel_cfg, proto.name, [DATE_A, DATE_B])

    run_prototype(panel_cfg, proto, [DATE_A, DATE_B], chunk_days=2, overwrite=True)
    p2 = load_explore_panel(panel_cfg, proto.name, [DATE_A, DATE_B])

    assert p1.sort(["date", "instrument", "ts_ms"]).equals(
        p2.sort(["date", "instrument", "ts_ms"])
    )


def test_run_prototype_missing_partition_raises_load_error(panel_cfg):
    proto = causal_proto()
    with pytest.raises(FileNotFoundError):
        load_explore_panel(panel_cfg, proto.name, [DATE_A, "20250699"])


def test_run_prototype_values_match_manual_computation(panel_cfg):
    """Runner output equals an independent manual recompute."""
    proto = causal_proto()
    run_prototype(panel_cfg, proto, [DATE_A], chunk_days=1)
    got = load_explore_panel(panel_cfg, proto.name, [DATE_A])

    from hft_autofactor.ingest import load_panel

    base = load_panel(panel_cfg, [DATE_A])
    exp_parts = []
    for part in base.sort(["date", "instrument", "ts_ms"]).partition_by(
        ["date", "instrument"], maintain_order=True
    ):
        vals = (
            part["mid_px"].log().diff(1).rolling_mean(window_size=5, min_samples=5)
        )
        exp_parts.append(part.select(["instrument", "ts_ms"]).with_columns(vals.alias(proto.name)))
    expected = pl.concat(exp_parts)

    joined = got.join(expected, on=["instrument", "ts_ms"], suffix="_ref")
    mismatch = joined.filter(
        ~(
            (pl.col(proto.name) == pl.col(f"{proto.name}_ref"))
            | (pl.col(proto.name).is_null() & pl.col(f"{proto.name}_ref").is_null())
        )
    )
    assert mismatch.is_empty()


def test_registry_used_by_runner_is_independent():
    """Registry collisions do not leak state between constructions."""
    r1 = PrototypeRegistry([causal_proto()])
    r2 = PrototypeRegistry([causal_proto()])
    assert r1.names() == r2.names() == ["causal_test_f"]
