"""Job discovery, CSV->parquet day builds, and panel loading (Stages 0/3/4 input).

The C++ engine emits one sorted CSV per (date, exchange, channel) under
``raw/{date}/{exchange}_ch{N}.csv``.  This module:

* discovers which jobs exist for a set of dates by scanning the read-only
  exchange data roots (instrument->channel mapping is re-discovered per day;
  everything downstream joins by InstrumentID, never channel);
* converts a day's channel CSVs into one parquet partition
  ``parquet/dt={date}/factors.parquet`` (asserting exactly one channel per
  instrument per day -- duplicates are errors, never silent merges);
* loads date blocks back as a single polars panel for evaluation.
"""
from __future__ import annotations

import re
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


def _factor_columns(columns: Sequence[str]) -> list[str]:
    labels = set(_label_columns(columns))
    return [
        c
        for c in columns
        if c not in BASE_COLUMNS and c not in labels and c != "channel"
    ]


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


def build_day_parquet(date: str, cfg: PipelineConfig, *, overwrite: bool = False) -> Path:
    """Merge all channel CSVs of one day into a single parquet partition.

    Asserts that each instrument appears in exactly ONE channel that day
    (the channel mapping changes across days, so joins are always by
    InstrumentID) and that (instrument, ts_ms) rows are unique.  Idempotent:
    an existing partition is returned unchanged unless ``overwrite``.
    """
    out_path = cfg.parquet_path(date)
    if out_path.exists() and not overwrite:
        return out_path

    raw_day_dir = cfg.raw_dir / date
    csvs = sorted(raw_day_dir.glob("*.csv")) if raw_day_dir.is_dir() else []
    if not csvs:
        raise FileNotFoundError(f"no raw channel CSVs under {raw_day_dir}")

    frames: list[pl.DataFrame] = []
    for csv_path in csvs:
        m = re.match(r"^(?P<ex>[a-z]+)_ch(?P<ch>\d+)\.csv$", csv_path.name)
        if not m:
            continue
        df = pl.read_csv(
            csv_path,
            null_values=["", "NaN", "nan"],
            # instrument codes ("510300") are pure digits but MUST stay
            # strings; every factor/label column is forced to Float64 so a
            # warm-up-only inference window cannot mis-type it as String
            schema_overrides=_interchange_schema_overrides(csv_path),
        )
        if df.is_empty():
            continue
        df = df.with_columns(pl.lit(int(m.group("ch")), dtype=pl.Int32).alias("channel"))
        frames.append(df)

    if not frames:
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
    """
    paths = [cfg.parquet_path(d) for d in dates]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing parquet partitions (run the convert stage first): "
            + ", ".join(missing)
        )

    df = pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")
    if instruments is not None:
        df = df.filter(pl.col("instrument").is_in(list(instruments)))

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
