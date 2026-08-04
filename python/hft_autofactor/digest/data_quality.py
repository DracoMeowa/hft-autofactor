"""Panel IO for the digest: lazy parquet scans, data quality, sampling.

Three responsibilities, all read-only against ``{out_root}/parquet``:

* :func:`parquet_paths_for_dates` -- resolve day partitions;
* :func:`panel_quality`           -- one streaming aggregation pass per
  partition producing the data-quality notes: flag-bit frequencies, ABSENT
  label rates, one-sided-book rates, and per-factor NaN rates overall and
  per exchange (feeds the taxonomy's "NaN-by-design on SSE" detection);
* :func:`sample_factor_rows`      -- stride-sampled factor columns for the
  correlation pass (never materializes a full production day).

Flag bit layout (cpp/include/hftaf/types.hpp, ``FlagBits``):
bit0 BOOK_UNSYNCED, bit1 SEQ_GAP_BEFORE, bit2 IOPV_INVALID,
bit3 ONE_SIDED_BOOK.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from ..ingest import BASE_COLUMNS, LABEL_PREFIXES

__all__ = [
    "FLAG_BIT_NAMES",
    "parquet_paths_for_dates",
    "panel_quality",
    "sample_factor_rows",
]

#: flag bit position -> engine name (types.hpp FlagBits)
FLAG_BIT_NAMES: dict[int, str] = {
    0: "book_unsynced",
    1: "seq_gap_before",
    2: "iopv_invalid",
    3: "one_sided_book",
}

_RESERVED = frozenset(BASE_COLUMNS) | {"channel"}


def parquet_paths_for_dates(out_root: str | Path, dates: Sequence[str]) -> list[Path]:
    """Existing ``parquet/dt={date}/factors.parquet`` paths, sorted by date."""
    root = Path(out_root)
    paths: list[Path] = []
    for d in sorted(set(dates)):
        p = root / "parquet" / f"dt={d}" / "factors.parquet"
        if p.is_file():
            paths.append(p)
    return paths


def factor_columns_of(columns: Sequence[str]) -> list[str]:
    """Factor columns = everything that is not base, channel, or a label."""
    out = []
    for c in columns:
        if c in _RESERVED:
            continue
        if any(c.startswith(p) for p in LABEL_PREFIXES):
            continue
        out.append(c)
    return out


def label_columns_of(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if any(c.startswith(p) for p in LABEL_PREFIXES)]


# --------------------------------------------------------------------- #
# data-quality aggregation                                              #
# --------------------------------------------------------------------- #
def _nan_expr(col: str):
    """Rows that are null OR NaN.

    Real panels store ABSENT/warm-up values as null (the CSV ''/NaN cells
    are mapped to null at convert time), but parquet partitions written by
    other tools may carry NaN floats -- count both encodings.
    """
    return pl.col(col).is_null() | pl.col(col).is_nan().fill_null(False)


def _quality_exprs(
    factor_cols: Sequence[str],
    label_cols: Sequence[str],
    *,
    have_flags: bool,
    have_quote: bool,
):
    exprs = [pl.len().alias("n_rows")]
    if have_flags:
        for bit, name in FLAG_BIT_NAMES.items():
            exprs.append(
                ((pl.col("flags") & (1 << bit)) != 0).sum().alias(f"flag_{name}")
            )
        exprs.append((pl.col("flags") == 0).sum().alias("flag_clean"))
    if have_quote:
        exprs.append(
            (
                pl.col("bid1_px").is_null()
                | (pl.col("bid1_px") <= 0)
                | pl.col("ask1_px").is_null()
                | (pl.col("ask1_px") <= 0)
            )
            .sum()
            .alias("quote_side_missing")
        )
    for f in factor_cols:
        exprs.append(_nan_expr(f).sum().alias(f"nan__{f}"))
    for lbl in label_cols:
        exprs.append(_nan_expr(lbl).sum().alias(f"absent__{lbl}"))
    return exprs


def _merge_counts(acc: dict, part: dict) -> dict:
    for k, v in part.items():
        if k == "exchange":
            continue
        acc[k] = acc.get(k, 0) + int(v or 0)
    return acc


def _collect_streaming(lf: "pl.LazyFrame") -> pl.DataFrame:
    """collect() preferring the streaming engine, version-tolerant.

    Newer polars takes ``engine="streaming"``, older polars takes
    ``streaming=True``; if the streaming engine cannot plan the query it
    falls back to the default in-memory engine (these queries only touch a
    handful of columns, so the fallback stays cheap).
    """
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # older polars: no ``engine`` parameter
        try:
            return lf.collect(streaming=True)
        except TypeError:  # ancient polars
            return lf.collect()
    except Exception:  # engine-specific planning failure -> in-memory
        return lf.collect()


def _schema_names(lf: "pl.LazyFrame") -> list[str]:
    try:
        return list(lf.collect_schema().names())
    except AttributeError:  # older polars
        return list(lf.schema.names())


def panel_quality(
    paths: Sequence[Path],
    *,
    factor_cols: Sequence[str] | None = None,
) -> dict:
    """Streaming data-quality pass over day partitions.

    Returns::

        {"n_rows", "n_partitions",
         "flag_bit_rates": {bit_name: rate}, "clean_rows_rate",
         "one_sided_book_rate",        # flag bit3 rate (engine's word)
         "quote_side_missing_rate",    # bid1/ask1 null-or<=0 cross-check
         "absent_label_rates": {label: rate},
         "factor_nan_rates": {factor: rate},
         "factor_nan_rates_by_exchange": {exchange: {factor: rate}},
         "n_rows_by_exchange": {exchange: int}}

    Factors missing from a partition are simply not counted there (rates are
    over rows where the column exists).
    """
    counts: dict = {}
    per_exchange: dict[str, dict] = {}
    factor_set = list(factor_cols) if factor_cols else None
    seen_factor_cols: list[str] = list(factor_cols) if factor_cols else []
    label_cols: list[str] = []
    n_partitions = 0

    for p in paths:
        lf = pl.scan_parquet(p)
        cols = _schema_names(lf)
        pf = (
            [f for f in factor_set if f in cols]
            if factor_set is not None
            else factor_columns_of(cols)
        )
        for f in pf:
            if f not in seen_factor_cols:
                seen_factor_cols.append(f)
        plabels = label_columns_of(cols)
        for lbl in plabels:
            if lbl not in label_cols:
                label_cols.append(lbl)

        have_flags = "flags" in cols
        have_quote = "bid1_px" in cols and "ask1_px" in cols
        exprs = _quality_exprs(
            pf, plabels, have_flags=have_flags, have_quote=have_quote
        )
        row = _collect_streaming(lf.select(exprs)).row(0, named=True)
        n_partitions += 1
        _merge_counts(counts, row)

        if "exchange" in cols:
            agg_exprs = [pl.len().alias("n_rows")] + [
                _nan_expr(f).sum().alias(f"nan__{f}") for f in pf
            ]
            gdf = _collect_streaming(
                lf.group_by("exchange").agg(agg_exprs)
            )
            for grow in gdf.iter_rows(named=True):
                exch = str(grow["exchange"])
                slot = per_exchange.setdefault(exch, {})
                _merge_counts(slot, grow)

    n_rows = int(counts.get("n_rows", 0))

    def rate(key: str) -> float:
        return float(counts.get(key, 0)) / n_rows if n_rows else float("nan")

    flag_bit_rates = {name: rate(f"flag_{name}") for name in FLAG_BIT_NAMES.values()}
    factor_nan_rates = {f: rate(f"nan__{f}") for f in seen_factor_cols}
    absent_label_rates = {lbl: rate(f"absent__{lbl}") for lbl in label_cols}

    nan_by_exchange: dict[str, dict] = {}
    n_rows_by_exchange: dict[str, int] = {}
    for exch, slot in per_exchange.items():
        en = int(slot.get("n_rows", 0))
        n_rows_by_exchange[exch] = en
        nan_by_exchange[exch] = {
            f: (float(slot.get(f"nan__{f}", 0)) / en if en else float("nan"))
            for f in seen_factor_cols
            if f"nan__{f}" in slot
        }

    return {
        "n_rows": n_rows,
        "n_partitions": n_partitions,
        "flag_bit_rates": flag_bit_rates,
        "clean_rows_rate": rate("flag_clean"),
        "one_sided_book_rate": flag_bit_rates.get("one_sided_book", float("nan")),
        "quote_side_missing_rate": rate("quote_side_missing"),
        "absent_label_rates": absent_label_rates,
        "factor_nan_rates": factor_nan_rates,
        "factor_nan_rates_by_exchange": nan_by_exchange,
        "n_rows_by_exchange": n_rows_by_exchange,
    }


# --------------------------------------------------------------------- #
# stride sampling for the correlation pass                              #
# --------------------------------------------------------------------- #
def sample_factor_rows(
    paths: Sequence[Path],
    factor_cols: Sequence[str],
    *,
    max_rows: int = 200_000,
) -> pl.DataFrame:
    """Stride-sampled factor matrix across partitions (<= ~max_rows rows).

    Each partition contributes at most ``ceil(max_rows / len(paths))`` rows
    taken on an even stride, so the sample spans the whole day (open, mid,
    close) rather than only the first snapshots.
    """
    if not paths or not factor_cols:
        return pl.DataFrame()
    per_part = max(1, -(-int(max_rows) // len(paths)))  # ceil division
    frames: list[pl.DataFrame] = []
    for p in paths:
        lf = pl.scan_parquet(p)
        cols = _schema_names(lf)
        keep = [f for f in factor_cols if f in cols]
        if not keep:
            continue
        lf = lf.select(keep)
        n = _collect_streaming(lf.select(pl.len())).item()
        if not n:
            continue
        step = max(1, n // per_part)
        idx = np.arange(0, n, step)[:per_part].tolist()
        frames.append(_collect_streaming(lf.gather(idx)))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")
