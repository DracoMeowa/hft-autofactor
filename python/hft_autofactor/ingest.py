"""Job discovery, CSV->parquet day builds, and panel loading (Stages 0/3/4 input).

The C++ engine emits one sorted CSV per (date, exchange, channel) under
``raw/{date}/{exchange}_ch{N}.csv``.  This module:

* discovers which jobs exist for a set of dates by scanning the read-only
  exchange data roots (instrument->channel mapping is re-discovered per day;
  everything downstream joins by InstrumentID, never channel);
* converts a day's channel CSVs into one parquet partition
  ``parquet/dt={date}/factors.parquet`` (asserting exactly one channel per
  instrument per day -- duplicates are errors, never silent merges).  The
  convert can be restricted to a set of instruments (single-instrument
  pilot); a ``factors.parquet.meta.json`` sidecar records the filter (and
  raw-CSV provenance) so the skip-if-done check never mistakes a filtered
  partition for a full one;
* loads date blocks back as a single polars panel for evaluation.  When an
  instrument filter is requested the load is lazy (``pl.scan_parquet`` with
  predicate pushdown) so unrequested instruments never reach RAM.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from .config import PipelineConfig

#: Channels that can contain ETFs (stocks/funds hash shards).  Channel 20 is
#: B-shares and 801 bonds -- never ETFs -- so they are excluded in v1.
ETF_CHANNELS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: The 12 canonical v1 factor columns, in registry order.
DEFAULT_FACTORS: tuple[str, ...] = (
    "quoted_spread_ticks",
    "microprice_dev",
    "oir",
    "wdi",
    "book_slope",
    "iopv_premium",
    "rv_60s",
    "rv_300s",
    "ofi_60s",
    "trade_imbalance_60s",
    "order_arrival_60s",
    "cancel_ratio_60s",
)

#: Canary look-ahead factors (only emitted by the engine with --canaries).
CANARY_FACTORS: tuple[str, ...] = ("future_mid_15s", "future_trade_sign")

#: Opt-in wishlist factor columns (NOT in DEFAULT_FACTORS so already-produced
#: runs keep their skip-if-done sidecars valid; materialized on demand via an
#: explicit ``factors:`` config list -- see docs/roadmap/panel-columns-wishlist.md).
#: Reserved as prototype names and available to explore-screen library dedup.
WISHLIST_FACTORS: tuple[str, ...] = (
    "avg_trade_size_60s",
    "n_trades_60s",
    "large_trade_share_60s",
    "trade_gap_ms",
    "cum_trade_vol",
)

#: Interchange columns that are always present and are not factor columns.
BASE_COLUMNS: tuple[str, ...] = (
    "date",
    "exchange",
    "instrument",
    "ts_ms",
    "snap_seq",
    "flags",
    "mid_px",
    "last_px",
    "bid1_px",
    "ask1_px",
    "bid1_qty",
    "ask1_qty",
    "depth_bid5",
    "depth_ask5",
)

#: Fixed dtypes of the interchange base columns. Every non-base column
#: (factors, labels) is float64. Declared explicitly because polars' default
#: 100-row inference window can fall entirely inside a factor's warm-up (or
#: an all-NaN column like the SSE cancel factors) and mis-infer it as String.
_BASE_DTYPES: dict[str, object] = {
    "date": pl.Utf8,
    "exchange": pl.Utf8,
    "instrument": pl.Utf8,
    "ts_ms": pl.Int64,
    "snap_seq": pl.Int64,
    "flags": pl.UInt32,
    "mid_px": pl.Float64,
    "last_px": pl.Float64,
    "bid1_px": pl.Float64,
    "ask1_px": pl.Float64,
    "bid1_qty": pl.Int64,
    "ask1_qty": pl.Int64,
    "depth_bid5": pl.Int64,
    "depth_ask5": pl.Int64,
}

#: Label column prefixes (per horizon suffix, e.g. fwd_mid_ret_15s).
LABEL_PREFIXES: tuple[str, ...] = ("fwd_mid_ret_", "fwd_last_ret_")

_CHANNEL_FILE_RE = re.compile(r"^1_channel_(\d+)\.csv\.gz$")

#: File-name pattern of interchange raw CSVs under ``raw/{date}/``.
_RAW_CSV_RE = re.compile(r"^(?P<ex>[a-z]+)_ch(?P<ch>\d+)\.csv$")


@dataclass(frozen=True)
class DayJob:
    """One engine job: a (date, exchange, channel) triple with its paths."""

    date: str
    exchange: str
    channel: int
    tick_gz: Path
    snapshot_gz: Path
    out_csv: Path


def _find_day_dir(root: Path, date: str) -> Path | None:
    """Locate the ``csv_MMDD_*`` directory of ``root/YYYYMM`` holding ``date``.

    If several dumps of the same day exist (different HHMMSS suffixes) the
    lexicographically last one (latest dump) wins.
    """
    month_dir = root / date[:6]
    if not month_dir.is_dir():
        return None
    mmdd = date[4:8]
    candidates = sorted(
        d
        for d in month_dir.glob(f"csv_{mmdd}*")
        if d.is_dir() and (d / "1_snapshot.csv.gz").is_file()
    )
    return candidates[-1] if candidates else None


def discover_jobs(cfg: PipelineConfig, dates: Sequence[str]) -> list[DayJob]:
    """Scan data roots and emit every runnable (date, exchange, channel) job.

    A job requires both ``1_channel_N.csv.gz`` (ticks) and ``1_snapshot.csv.gz``
    to exist.  Only ETF-bearing channels (1..6) are returned.
    """
    jobs: list[DayJob] = []
    for date in dates:
        for exchange, root in cfg.data_roots.items():
            day_dir = _find_day_dir(Path(root), date)
            if day_dir is None:
                continue
            snapshot_gz = day_dir / "1_snapshot.csv.gz"
            for entry in sorted(day_dir.iterdir()):
                m = _CHANNEL_FILE_RE.match(entry.name)
                if not m:
                    continue
                channel = int(m.group(1))
                if channel not in ETF_CHANNELS:
                    continue
                jobs.append(
                    DayJob(
                        date=date,
                        exchange=exchange,
                        channel=channel,
                        tick_gz=entry,
                        snapshot_gz=snapshot_gz,
                        out_csv=cfg.raw_csv(date, exchange, channel),
                    )
                )
    return jobs


def _label_columns(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if any(c.startswith(p) for p in LABEL_PREFIXES)]


def factor_columns(columns: Sequence[str]) -> list[str]:
    """The factor columns among ``columns`` (not base, not labels, not channel)."""
    labels = set(_label_columns(columns))
    return [
        c
        for c in columns
        if c not in BASE_COLUMNS and c not in labels and c != "channel"
    ]


#: Backwards-compatible alias (was the private name).
_factor_columns = factor_columns


def _interchange_schema_overrides(csv_path: Path) -> dict:
    """Explicit dtypes for every column of one interchange CSV.

    Reads only the header row. Base columns take their fixed dtypes; every
    other column (factor or label) is Float64. This is REQUIRED: polars'
    default 100-row inference window can fall entirely inside a factor's
    warm-up (or an all-NaN column such as the SSE cancel factors) and would
    then mis-infer the column as String.
    """
    with open(csv_path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\r\n")
    overrides = dict(_BASE_DTYPES)
    for col in header.split(","):
        col = col.strip()
        if col and col not in overrides:
            overrides[col] = pl.Float64
    return overrides


def convert_meta_path(out_path: Path) -> Path:
    """Sidecar path for one day partition: ``factors.parquet.meta.json``."""
    return out_path.parent / (out_path.name + ".meta.json")


def _raw_channel_csvs(raw_day_dir: Path) -> list[Path]:
    """Sorted interchange raw CSVs (``{exchange}_ch{N}.csv``) of one day."""
    if not raw_day_dir.is_dir():
        return []
    return sorted(
        p for p in raw_day_dir.glob("*.csv") if _RAW_CSV_RE.match(p.name)
    )


def _convert_uptodate(
    date: str, cfg: PipelineConfig, instruments: list[str] | None
) -> bool:
    """Skip-if-done for the convert stage.

    An existing partition is reusable iff it COVERS the requested filter:

    * a partition recorded as full (sidecar ``instruments: null``) covers
      every request;
    * a legacy partition WITHOUT a sidecar predates the filter and was
      always a full panel, so it also covers every request;
    * a filtered partition covers only requests whose instrument set is a
      subset of the recorded filter (a full-panel request therefore
      invalidates it and forces a rebuild).

    When a sidecar is present the recorded raw-CSV provenance (file set +
    sizes) must also match the current ``raw/{date}`` directory, so changed
    or newly added engine outputs invalidate the partition.
    """
    out_path = cfg.parquet_path(date)
    if not out_path.is_file():
        return False
    meta_path = convert_meta_path(out_path)
    if not meta_path.is_file():
        return True  # legacy partition: always a full panel

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    recorded = meta.get("instruments")
    if recorded is not None:
        if instruments is None:
            return False  # filtered partition cannot satisfy a full request
        if not set(map(str, recorded)) >= set(instruments):
            return False

    sizes = meta.get("source_csvs")
    if isinstance(sizes, dict):
        try:
            recorded_sizes = {str(k): int(v) for k, v in sizes.items()}
        except (TypeError, ValueError):
            return False
        current = {
            p.name: p.stat().st_size for p in _raw_channel_csvs(cfg.raw_dir / date)
        }
        if recorded_sizes != current:
            return False
    return True


def build_day_parquet(
    date: str,
    cfg: PipelineConfig,
    *,
    overwrite: bool = False,
    instruments: Sequence[str] | None = None,
) -> Path:
    """Merge all channel CSVs of one day into a single parquet partition.

    Asserts that each instrument appears in exactly ONE channel that day
    (the channel mapping changes across days, so joins are always by
    InstrumentID) and that (instrument, ts_ms) rows are unique.  Idempotent:
    an existing partition covering the requested ``instruments`` filter is
    returned unchanged unless ``overwrite``.

    ``instruments`` restricts the partition to those instrument codes
    (single-instrument pilot).  The effective filter is recorded in the
    ``factors.parquet.meta.json`` sidecar next to the partition so a
    filtered parquet is never mistaken for a full one by the skip-if-done
    check (see :func:`_convert_uptodate`).  With a filter active the CSVs
    are read lazily so rows of other instruments never reach RAM.
    """
    wanted: list[str] | None = (
        list(dict.fromkeys(str(i) for i in instruments))
        if instruments is not None
        else None
    )
    out_path = cfg.parquet_path(date)
    if not overwrite and _convert_uptodate(date, cfg, wanted):
        return out_path

    raw_day_dir = cfg.raw_dir / date
    csvs = _raw_channel_csvs(raw_day_dir)
    if not csvs:
        raise FileNotFoundError(f"no raw channel CSVs under {raw_day_dir}")

    frames: list[pl.DataFrame] = []
    for csv_path in csvs:
        m = _RAW_CSV_RE.match(csv_path.name)
        if not m:
            continue
        # instrument codes ("510300") are pure digits but MUST stay
        # strings; every factor/label column is forced to Float64 so a
        # warm-up-only inference window cannot mis-type it as String
        overrides = _interchange_schema_overrides(csv_path)
        if wanted is None:
            df = pl.read_csv(
                csv_path,
                null_values=["", "NaN", "nan"],
                schema_overrides=overrides,
            )
        else:
            # lazy scan: the instrument filter is pushed down into the CSV
            # reader, so rows of unrequested instruments are never built
            df = (
                pl.scan_csv(
                    csv_path,
                    null_values=["", "NaN", "nan"],
                    schema_overrides=overrides,
                )
                .filter(pl.col("instrument").is_in(wanted))
                .collect()
            )
        if df.is_empty():
            continue
        df = df.with_columns(pl.lit(int(m.group("ch")), dtype=pl.Int32).alias("channel"))
        frames.append(df)

    if not frames:
        if wanted is not None:
            raise ValueError(
                f"day {date}: no rows for instruments {wanted} in raw CSVs "
                f"under {raw_day_dir}"
            )
        raise FileNotFoundError(f"raw CSVs under {raw_day_dir} contain no rows")

    panel = pl.concat(frames, how="vertical_relaxed")

    # --- integrity checks: mark the day suspect instead of silent merges ---
    dup_ch = (
        panel.group_by("instrument")
        .agg(pl.col("channel").n_unique().alias("n_ch"))
        .filter(pl.col("n_ch") > 1)
    )
    if not dup_ch.is_empty():
        bad = ", ".join(sorted(dup_ch["instrument"].cast(str).unique().to_list()))
        raise ValueError(
            f"day {date}: instruments present in more than one channel: {bad}"
        )

    dup_rows = (
        panel.group_by("instrument", "ts_ms")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if not dup_rows.is_empty():
        raise ValueError(
            f"day {date}: duplicate (instrument, ts_ms) rows across channel files"
        )

    panel = panel.sort(["instrument", "ts_ms"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out_path)

    meta = {
        "date": date,
        # null => full panel; list => converted with that instrument filter
        "instruments": wanted,
        "full_panel": wanted is None,
        "n_rows": int(panel.height),
        "source_csvs": {p.name: p.stat().st_size for p in csvs},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    convert_meta_path(out_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


def load_panel(
    cfg: PipelineConfig,
    dates: Sequence[str],
    *,
    instruments: Sequence[str] | None = None,
    factors: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Load day partitions for ``dates`` into one panel.

    Columns kept: base columns + ``channel`` + selected factor columns +
    all label columns.  ``factors=None`` keeps every factor column found.

    With ``instruments`` given, each partition is read LAZILY via
    ``pl.scan_parquet`` and the instrument filter is applied as a predicate
    (pushed down into the parquet scan), so rows of unrequested instruments
    never materialize in RAM -- important when the partitions hold the full
    market but only a single-instrument pilot panel is needed.  Without a
    filter the partitions are read eagerly as before.
    """
    paths = [cfg.parquet_path(d) for d in dates]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing parquet partitions (run the convert stage first): "
            + ", ".join(missing)
        )

    if instruments is not None:
        wanted = list(dict.fromkeys(str(i) for i in instruments))
        frames = [
            pl.scan_parquet(p)
            .filter(pl.col("instrument").is_in(wanted))
            .collect()
            for p in paths
        ]
        df = pl.concat(frames, how="vertical_relaxed")
    else:
        df = pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")

    if instruments is not None and df.is_empty():
        raise ValueError(
            f"no rows for instruments {list(instruments)} in partitions: "
            + ", ".join(str(p) for p in paths)
        )

    keep = [c for c in BASE_COLUMNS if c in df.columns]
    if "channel" in df.columns:
        keep.append("channel")
    label_cols = _label_columns(df.columns)
    keep.extend(label_cols)

    available_factors = _factor_columns(df.columns)
    if factors is None:
        keep.extend(available_factors)
    else:
        wanted = [f for f in factors if f in available_factors]
        keep.extend(wanted)

    return df.select([c for c in dict.fromkeys(keep)])
