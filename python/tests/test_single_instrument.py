"""Single-instrument panel support (the 588000 pilot).

The full-market pipeline assumed a real cross-section of ETFs; with ONE
instrument several code paths break or become meaningless:

* ``load_panel`` must lazy-scan and push the instrument predicate down
  instead of reading every partition fully into RAM;
* ``build_day_parquet`` must support an instrument filter and record it in a
  sidecar so skip-if-done never mistakes a filtered parquet for a full one;
* ``run_eval_stage`` must skip cross-sectional IC (undefined for 1
  instrument) and base all gating on the per-(date) time-series RankIC;
* ``permutation_noise_floor`` must shuffle labels within (date) blocks.

Each behaviour gets a dedicated test here on synthetic 1-instrument panels.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from conftest import HORIZONS, make_day_rows, write_interchange_csv

from hft_autofactor.config import PipelineConfig
from hft_autofactor.eval.gating import permutation_noise_floor
from hft_autofactor.eval.ic import ic_stats, rank_ic_time_series
from hft_autofactor.ingest import build_day_parquet, convert_meta_path, load_panel
from hft_autofactor.pipeline import orchestrator

DATE = "20250603"
DATES = [f"202506{d:02d}" for d in range(3, 15)]  # 12 consecutive days
INST = "588000"
OTHER = "510300"

#: snapshots per day for eval-scale panels: > 900s/3s = 300 so every horizon
#: keeps non-empty labels (ABSENT semantics only at the day tail)
N_SNAP_EVAL = 360


# --------------------------------------------------------------------- #
# synthetic fixtures                                                    #
# --------------------------------------------------------------------- #
def _write_raw_day(
    cfg: PipelineConfig, date: str, instruments, channel: int = 1, n_snap: int = 12
):
    """Write one channel CSV holding ``instruments`` for ``date``."""
    rows = []
    for inst in instruments:
        rows.extend(make_day_rows(inst, n_snap=n_snap, factors=("oir", "wdi")))
    write_interchange_csv(
        cfg.raw_csv(date, "sse", channel),
        date=date, exchange="sse", rows=rows, factors=("oir", "wdi"),
    )


def make_noisy_rows(instrument, *, n_snap, seed, start_ms=34_200_000,
                    step_ms=3000, noise_sigma=0.5):
    """Rows with a REAL but imperfect signal: oir_t = z, every fwd label of a
    valid row = z + sigma*noise (time-series RankIC ~ 0.9, non-degenerate
    across days); wdi is pure noise (no signal)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_snap):
        ts = start_ms + i * step_ms
        z = float(rng.standard_normal())
        r = {
            "instrument": instrument,
            "ts_ms": ts,
            "snap_seq": 1000 + i,
            "flags": 0,
            "mid_px": 1.28,
            "last_px": 1.28,
            "bid1_px": 1.279,
            "ask1_px": 1.281,
            "bid1_qty": 10000,
            "ask1_qty": 8000,
            "depth_bid5": 50000,
            "depth_ask5": 42000,
            "oir": z,
            "wdi": float(rng.standard_normal()),
        }
        for h in HORIZONS:
            ahead = i + (h * 1000) // step_ms
            value = None if ahead >= n_snap else z + noise_sigma * float(
                rng.standard_normal()
            )
            r[f"fwd_mid_ret_{h}s"] = value
            r[f"fwd_last_ret_{h}s"] = value
        rows.append(r)
    return rows


def _write_noisy_day(cfg: PipelineConfig, date: str, instrument, seed):
    rows = make_noisy_rows(instrument, n_snap=N_SNAP_EVAL, seed=seed)
    write_interchange_csv(
        cfg.raw_csv(date, "sse", 1),
        date=date, exchange="sse", rows=rows, factors=("oir", "wdi"),
    )


def _build_noisy_partitions(cfg: PipelineConfig, dates, instrument=INST):
    for i, d in enumerate(dates):
        _write_noisy_day(cfg, d, instrument, seed=100 + i)
        build_day_parquet(d, cfg, instruments=[instrument])


# --------------------------------------------------------------------- #
# load_panel: lazy instrument filter                                    #
# --------------------------------------------------------------------- #
def test_load_panel_instrument_filter_rows_and_columns(small_cfg):
    _write_raw_day(small_cfg, DATE, [INST, OTHER])
    build_day_parquet(DATE, small_cfg)

    sub = load_panel(small_cfg, [DATE], instruments=[INST])
    assert set(sub["instrument"].unique().to_list()) == {INST}
    assert sub.height == 12  # only the requested instrument's 12 snaps

    full = load_panel(small_cfg, [DATE])
    assert full.height == 24  # both instruments
    # identical column set either way (the filter only restricts rows)
    assert sub.columns == full.columns


def test_load_panel_instrument_filter_is_lazy(small_cfg, monkeypatch):
    """With an instrument filter the panel must be read via a lazy scan with
    predicate pushdown, never by eagerly loading each full partition."""
    _write_raw_day(small_cfg, DATE, [INST, OTHER])
    build_day_parquet(DATE, small_cfg)

    import hft_autofactor.ingest as ingest_mod

    eager_calls: list = []
    real_read_parquet = pl.read_parquet

    def spy_read_parquet(*a, **k):
        eager_calls.append(a)
        return real_read_parquet(*a, **k)

    lazy_calls: list = []
    real_scan_parquet = pl.scan_parquet

    def spy_scan_parquet(*a, **k):
        lazy_calls.append(a)
        return real_scan_parquet(*a, **k)

    monkeypatch.setattr(ingest_mod.pl, "read_parquet", spy_read_parquet)
    monkeypatch.setattr(ingest_mod.pl, "scan_parquet", spy_scan_parquet)

    sub = load_panel(small_cfg, [DATE], instruments=[INST])
    assert set(sub["instrument"].unique().to_list()) == {INST}
    # the filtered load went through the lazy scanner, not the eager reader
    assert len(lazy_calls) == 1
    assert eager_calls == []


def test_load_panel_multi_day_instrument_filter(small_cfg):
    for d in DATES[:3]:
        _write_raw_day(small_cfg, d, [INST, OTHER])
        build_day_parquet(d, small_cfg)
    sub = load_panel(small_cfg, DATES[:3], instruments=[INST])
    assert set(sub["instrument"].unique().to_list()) == {INST}
    assert set(sub["date"].unique().to_list()) == set(DATES[:3])
    assert sub.height == 3 * 12


def test_load_panel_instrument_filter_no_rows_raises(small_cfg):
    _write_raw_day(small_cfg, DATE, [INST])
    build_day_parquet(DATE, small_cfg)
    with pytest.raises(ValueError, match="no rows for instruments"):
        load_panel(small_cfg, [DATE], instruments=["999999"])


# --------------------------------------------------------------------- #
# build_day_parquet: instrument filter + sidecar                        #
# --------------------------------------------------------------------- #
def test_build_day_parquet_instrument_filter_and_sidecar(small_cfg):
    _write_raw_day(small_cfg, DATE, [INST, OTHER])
    out = build_day_parquet(DATE, small_cfg, instruments=[INST])
    df = pl.read_parquet(out)
    assert set(df["instrument"].unique().to_list()) == {INST}

    meta_path = convert_meta_path(out)
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["instruments"] == [INST]
    assert meta["full_panel"] is False
    assert meta["n_rows"] == df.height
    assert set(meta["source_csvs"]) == {"sse_ch1.csv"}


def test_build_day_parquet_full_sidecar_records_no_filter(small_cfg):
    _write_raw_day(small_cfg, DATE, [INST, OTHER])
    out = build_day_parquet(DATE, small_cfg)
    meta = json.loads(convert_meta_path(out).read_text(encoding="utf-8"))
    assert meta["instruments"] is None
    assert meta["full_panel"] is True


def test_build_day_parquet_sidecar_invalidated_on_filter_change(small_cfg):
    _write_raw_day(small_cfg, DATE, [INST, OTHER])

    # start from a filtered partition
    p_filt = build_day_parquet(DATE, small_cfg, instruments=[INST])
    assert set(pl.read_parquet(p_filt)["instrument"].unique().to_list()) == {INST}

    # same filter -> reused, not rewritten
    mtime = p_filt.stat().st_mtime_ns
    build_day_parquet(DATE, small_cfg, instruments=[INST])
    assert p_filt.stat().st_mtime_ns == mtime

    # a DIFFERENT filter is not covered by [INST] -> rebuild
    p_other = build_day_parquet(DATE, small_cfg, instruments=[OTHER])
    assert p_other == p_filt  # same canonical path, rebuilt in place
    assert set(pl.read_parquet(p_other)["instrument"].unique().to_list()) == {OTHER}
    meta_other = json.loads(convert_meta_path(p_other).read_text(encoding="utf-8"))
    assert meta_other["instruments"] == [OTHER]

    # request the FULL panel -> a filtered partition must NOT be reused
    # (this is the confusion the sidecar exists to prevent)
    p_full = build_day_parquet(DATE, small_cfg)
    assert set(pl.read_parquet(p_full)["instrument"].unique().to_list()) == {INST, OTHER}
    meta = json.loads(convert_meta_path(p_full).read_text(encoding="utf-8"))
    assert meta["instruments"] is None
    assert meta["full_panel"] is True


def test_build_day_parquet_full_partition_covers_filtered_request(small_cfg):
    """A FULL partition satisfies a filtered request (the reverse direction is
    safe -- consumers filter at load), so it is reused, never replaced by a
    smaller partition."""
    _write_raw_day(small_cfg, DATE, [INST, OTHER])
    out = build_day_parquet(DATE, small_cfg)
    mtime = out.stat().st_mtime_ns

    reused = build_day_parquet(DATE, small_cfg, instruments=[INST])
    assert reused == out
    assert out.stat().st_mtime_ns == mtime  # skipped, not rebuilt
    # partition stayed full and the sidecar still records a full panel
    assert set(pl.read_parquet(out)["instrument"].unique().to_list()) == {INST, OTHER}
    meta = json.loads(convert_meta_path(out).read_text(encoding="utf-8"))
    assert meta["instruments"] is None and meta["full_panel"] is True


def test_build_day_parquet_sidecar_invalidated_on_input_change(small_cfg):
    _write_raw_day(small_cfg, DATE, [INST, OTHER])
    out = build_day_parquet(DATE, small_cfg, instruments=[INST])
    meta_path = convert_meta_path(out)
    before = json.loads(meta_path.read_text(encoding="utf-8"))

    # change the inputs: a second channel CSV appears (e.g. backfilled)
    extra = make_day_rows("159915", n_snap=12, factors=("oir", "wdi"))
    write_interchange_csv(
        small_cfg.raw_csv(DATE, "sse", 2), date=DATE, exchange="sse",
        rows=extra, factors=("oir", "wdi"),
    )
    # same filter, but the input set changed -> must rebuild
    build_day_parquet(DATE, small_cfg, instruments=[INST])
    after = json.loads(meta_path.read_text(encoding="utf-8"))
    assert set(after["source_csvs"]) == {"sse_ch1.csv", "sse_ch2.csv"}
    assert after["source_csvs"] != before["source_csvs"]
    # the filtered content stayed filtered across the rebuild
    assert set(pl.read_parquet(out)["instrument"].unique().to_list()) == {INST}


def test_build_day_parquet_filter_no_matching_rows_raises(small_cfg):
    _write_raw_day(small_cfg, DATE, [INST])
    with pytest.raises(ValueError, match="no rows for instruments"):
        build_day_parquet(DATE, small_cfg, instruments=["999999"])


def test_build_day_parquet_legacy_partition_without_sidecar_reused(small_cfg):
    """A pre-sidecar partition (no .meta.json) is a full panel by history and
    must be reused as-is, not rebuilt just because the sidecar is missing."""
    _write_raw_day(small_cfg, DATE, [INST, OTHER])
    out = build_day_parquet(DATE, small_cfg)
    convert_meta_path(out).unlink()  # simulate a legacy partition
    mtime = out.stat().st_mtime_ns
    build_day_parquet(DATE, small_cfg)  # full request -> reuse legacy
    assert out.stat().st_mtime_ns == mtime
    # a filtered request is also covered by the full legacy partition
    build_day_parquet(DATE, small_cfg, instruments=[INST])
    assert out.stat().st_mtime_ns == mtime


# --------------------------------------------------------------------- #
# eval stage end-to-end on ONE instrument                               #
# --------------------------------------------------------------------- #
def test_eval_stage_single_instrument_skips_cross_section(small_cfg):
    _build_noisy_partitions(small_cfg, DATES)

    report_path = orchestrator.run_eval_stage(small_cfg, DATES, instruments=[INST])
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["n_instruments"] == 1
    assert report["instruments"] == [INST]
    assert report["cross_section_skipped"] is True
    assert report["cross_section_stats"] == []
    # gating still produced time-series stats + a screen for every pair
    assert len(report["stats"]) == 2 * len(HORIZONS)  # oir + wdi x 5 horizons
    assert len(report["stage1_screen"]) == len(report["stats"])
    for st in report["stats"]:
        assert st["n_obs"] == len(DATES)  # one time-series IC per day


def test_eval_stage_single_instrument_time_series_gating(small_cfg):
    """On ONE instrument the Stage-1 screen and the permutation floor must run
    on the time-series RankIC and still separate signal from noise."""
    _build_noisy_partitions(small_cfg, DATES)

    report_path = orchestrator.run_eval_stage(small_cfg, DATES, instruments=[INST])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    screen = {(r["factor"], r["horizon_s"]): r for r in report["stage1_screen"]}
    oir60 = screen[("oir", 60)]
    # oir carries the planted time-series signal -> passes on TS stats alone
    assert oir60["mean_ic"] > 0.8
    assert oir60["t_stat_nw"] >= oir60["t_hurdle_min"]
    assert oir60["passed"] is True

    # the noise floor was computed on the 1-instrument panel and is finite
    floors = {
        (f["factor"], f["horizon_s"]): f["floor"] for f in report["noise_floors"]
    }
    assert floors[("oir", 60)] >= 0.0
    assert oir60["mean_ic"] > floors[("oir", 60)]

    # walk-forward folds ran on the time-series stats for every pair
    assert len(report["walk_forward"]) > 0
    oir60_folds = [
        w for w in report["walk_forward"]
        if w["factor"] == "oir" and w["horizon_s"] == 60
    ]
    assert len(oir60_folds) >= 1
    assert all(w["passed"] for w in oir60_folds)


def test_eval_stage_cross_section_enabled_with_enough_instruments(small_cfg):
    """Backwards compat: with >= 5 instruments cross-sectional IC still runs."""
    instruments = [INST, OTHER, "510050", "159915", "512100"]
    dates = DATES[:3]
    for i, d in enumerate(dates):
        rows = []
        for j, inst in enumerate(instruments):
            # distinct seed per instrument: identical seeds would align all
            # cross-section values and make cross-sectional IC degenerate
            rows.extend(
                make_noisy_rows(inst, n_snap=N_SNAP_EVAL, seed=200 + 100 * i + 7 * j)
            )
        write_interchange_csv(
            small_cfg.raw_csv(d, "sse", 1), date=d, exchange="sse",
            rows=rows, factors=("oir", "wdi"),
        )
        build_day_parquet(d, small_cfg)

    report = json.loads(
        orchestrator.run_eval_stage(small_cfg, dates).read_text(encoding="utf-8")
    )
    assert report["n_instruments"] == 5
    assert report["cross_section_skipped"] is False
    assert len(report["cross_section_stats"]) == len(report["stats"])
    # cross-sectional ICs are actually populated (5 aligned instruments)
    assert any(st["n_obs"] > 0 for st in report["cross_section_stats"])

    # 2 instruments stay below the threshold -> still skipped
    dates2 = DATES[3:5]
    for i, d in enumerate(dates2):
        rows = make_noisy_rows(INST, n_snap=N_SNAP_EVAL, seed=300 + i)
        rows.extend(make_noisy_rows(OTHER, n_snap=N_SNAP_EVAL, seed=400 + i))
        write_interchange_csv(
            small_cfg.raw_csv(d, "sse", 1), date=d, exchange="sse",
            rows=rows, factors=("oir", "wdi"),
        )
        build_day_parquet(d, small_cfg)
    report2 = json.loads(
        orchestrator.run_eval_stage(small_cfg, dates2).read_text(encoding="utf-8")
    )
    assert report2["n_instruments"] == 2
    assert report2["cross_section_skipped"] is True
    assert report2["cross_section_stats"] == []


# --------------------------------------------------------------------- #
# permutation noise floor on ONE instrument                             #
# --------------------------------------------------------------------- #
def _one_inst_panel(days, n_rows=40, noise=0.05, seed=11):
    rng = np.random.default_rng(seed)
    records = []
    for d in days:
        for r in range(n_rows):
            f = rng.standard_normal()
            y = f + noise * rng.standard_normal()
            records.append(
                {
                    "date": d,
                    "instrument": INST,
                    "ts_ms": 34_200_000 + r * 3000,
                    "oir": f,
                    "fwd_mid_ret_60s": y,
                }
            )
    return pl.DataFrame(records)


def test_permutation_noise_floor_single_instrument_finite():
    panel = _one_inst_panel(DATES[:5])
    floor = permutation_noise_floor(panel, "oir", 60, n_perms=20)
    assert floor >= 0.0
    assert floor < 1.0  # a real signal's |mean IC| can exceed it


def test_permutation_noise_floor_shuffles_within_date_blocks(monkeypatch):
    """For a 1-instrument panel the (date, instrument) blocks collapse to
    (date) blocks: every permutation must keep each day's label multiset
    inside that day -- labels may never leak across days."""
    import hft_autofactor.eval.gating as gating_mod

    panel = _one_inst_panel(DATES[:3])
    expected: dict[str, list[float]] = {}
    for part in panel.select(["date", "fwd_mid_ret_60s"]).partition_by("date"):
        expected[part["date"][0]] = sorted(part["fwd_mid_ret_60s"].to_list())

    real_rank_ic = gating_mod.rank_ic_time_series
    captured: list[pl.DataFrame] = []

    def spy(panel_arg, factor, horizon_s, label="fwd_mid_ret"):
        captured.append(panel_arg)
        return real_rank_ic(panel_arg, factor, horizon_s, label=label)

    monkeypatch.setattr(gating_mod, "rank_ic_time_series", spy)
    n_perms = 5
    floor = gating_mod.permutation_noise_floor(panel, "oir", 60, n_perms=n_perms)
    monkeypatch.undo()

    assert floor >= 0.0
    assert len(captured) == n_perms
    for perm_panel in captured:
        # no rows created or lost by the shuffle
        assert perm_panel.height == panel.height
        for part in perm_panel.select(["date", "fwd_mid_ret_60s"]).partition_by("date"):
            d = part["date"][0]
            assert sorted(part["fwd_mid_ret_60s"].to_list()) == expected[d]


def test_permutation_floor_below_true_signal_single_instrument():
    """A strong signal's time-series |mean IC| must clear the 1-instrument
    permutation floor: the gate stays meaningful with one instrument."""
    panel = _one_inst_panel(DATES[:6], noise=0.05, seed=3)
    floor = permutation_noise_floor(panel, "oir", 60, n_perms=30)
    st = ic_stats(rank_ic_time_series(panel, "oir", 60), "oir", 60)
    assert abs(st.mean_ic) > floor
    assert st.n_obs == 6  # one time-series IC per day
