"""Derived factors for the backtest: recomputed from panel base columns.

iter-001 admitted prototype factors (explore lane) that are NOT materialized
in the parquet panel.  Each formula below reproduces the admitted explore
spec exactly (same window constants, same warm-up semantics), computed
independently per ``(instrument, date)`` group so no state leaks across day
boundaries -- matching the engine contract that the channel/instrument
mapping changes daily.

Registered factors
------------------
* ``depth5_delta_60s`` -- trailing-60s change (delta) of 5-level depth
  imbalance ``(bid - ask) / (bid + ask)``; 20 x 3s rows.  Needs only the
  base columns ``depth_bid5`` / ``depth_ask5``.
* ``flow_divergence_300s`` -- ``z300(ofi_60s) - z300(trade_imbalance_60s)``
  with a trailing-300s (100-row) z-score; constant windows map to 0.0.
  Needs the library factor columns ``ofi_60s`` and ``trade_imbalance_60s``.

Materializing here (instead of demanding new parquet columns) keeps the
backtest reproducible from the existing interchange panel and lets the
C++-promotion validation (report 7.1/7.2) compare against one canonical
Python reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import polars as pl

__all__ = [
    "DerivedFactor",
    "DERIVED_FACTORS",
    "is_derived",
    "materialize_derived",
]

#: Rows per 60s on the 3s snapshot grid.
_ROWS_60S = 20
#: Rows per 300s on the 3s snapshot grid (also the z-score window).
_ROWS_300S = 100


@dataclass(frozen=True)
class DerivedFactor:
    """One registered derived factor.

    ``sources`` are the panel factor columns that must be loaded for the
    derivation (base interchange columns are always loaded).  ``compute``
    receives one ``(instrument, date)`` group SORTED BY ``ts_ms`` and
    returns the factor values as a Series aligned with that group.
    """

    name: str
    sources: tuple[str, ...]
    compute: Callable[[pl.DataFrame], pl.Series]
    doc: str


def _depth5_delta_60s(part: pl.DataFrame) -> pl.Series:
    """Delta over 20 rows (60s) of 5-level depth imbalance.

    Mirrors ``explore-specs-iter001/depth5_delta_60s.py`` byte-for-byte:
    imbalance is null when the total depth is <= 0; ``diff(20)`` leaves the
    first 20 rows of each day null (warm-up, never zero-filled).
    """
    b = pl.col("depth_bid5").cast(pl.Float64)
    a = pl.col("depth_ask5").cast(pl.Float64)
    tot = b + a
    imb = (
        pl.when(tot > 0.0)
        .then((b - a) / tot)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(imb.diff(_ROWS_60S).alias("value"))["value"]


def _flow_divergence_300s(part: pl.DataFrame) -> pl.Series:
    """z300(OFI@60s) - z300(TI@60s), trailing-300s z-scores.

    Mirrors ``explore-specs-iter001/flow_divergence_300s.py``: rolling
    mean/std over 100 rows with ``min_samples=100`` (first 99 rows null),
    constant windows (std == 0) mapped to 0.0 (neutral).
    """
    w = _ROWS_300S

    def z300(x: pl.Expr) -> pl.Expr:
        m = x.rolling_mean(window_size=w, min_samples=w)
        s = x.rolling_std(window_size=w, min_samples=w)
        z = (x - m) / s
        return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)

    ofi = pl.col("ofi_60s")
    ti = pl.col("trade_imbalance_60s")
    return part.select((z300(ofi) - z300(ti)).alias("value"))["value"]


DERIVED_FACTORS: dict[str, DerivedFactor] = {
    "depth5_delta_60s": DerivedFactor(
        name="depth5_delta_60s",
        sources=(),
        compute=_depth5_delta_60s,
        doc=(
            "trailing-60s delta of 5-level depth imbalance "
            "(depth_bid5/depth_ask5 base columns only)"
        ),
    ),
    "flow_divergence_300s": DerivedFactor(
        name="flow_divergence_300s",
        sources=("ofi_60s", "trade_imbalance_60s"),
        compute=_flow_divergence_300s,
        doc="z300(ofi_60s) - z300(trade_imbalance_60s), 100-row trailing z",
    ),
}


def is_derived(factor: str) -> bool:
    """True when ``factor`` is a registered derived factor."""
    return factor in DERIVED_FACTORS


def materialize_derived(panel: pl.DataFrame, factor: str) -> pl.DataFrame:
    """Append the derived ``factor`` column to ``panel``.

    Computed per ``(instrument, date)`` group on rows sorted by ``ts_ms``;
    the result is realigned to the input row order via a row index, so the
    input frame's ordering is preserved.  Source columns must already be
    present (load them via ``DerivedFactor.sources``).
    """
    spec = DERIVED_FACTORS.get(factor)
    if spec is None:
        raise KeyError(f"unknown derived factor {factor!r}")
    missing = [c for c in spec.sources if c not in panel.columns]
    if missing:
        raise ValueError(
            f"derived factor {factor!r} needs columns {missing} which are "
            "missing from the panel"
        )
    for col in ("instrument", "date", "ts_ms"):
        if col not in panel.columns:
            raise ValueError(
                f"materialize_derived: panel lacks grouping column {col!r}"
            )

    if panel.height == 0:
        return panel.with_columns(
            pl.Series(factor, [], dtype=pl.Float64)
        )

    indexed = panel.with_row_index("__hftaf_row")
    ordered = indexed.sort(["instrument", "date", "ts_ms"])
    chunks: list[pl.DataFrame] = []
    for _key, group in ordered.group_by(
        ["instrument", "date"], maintain_order=True
    ):
        values = spec.compute(group)
        chunks.append(
            pl.DataFrame(
                {"__hftaf_row": group["__hftaf_row"], factor: values}
            )
        )
    out = pl.concat(chunks).sort("__hftaf_row")
    return indexed.drop("__hftaf_row").with_columns(out[factor])
