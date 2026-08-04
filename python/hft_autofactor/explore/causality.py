"""Panel prefix causality test: truncate-and-recompute on the parquet panel.

The panel-level analogue of ``validation/mask_test.py`` (MASK TEST A).  The
engine-level test truncates raw streams and reruns the C++ binary; here the
"binary" is the prototype's compute spec and the stream is the panel itself:

  1. compute the prototype on the FULL panel -> full column;
  2. choose K truncation cutoffs T spread over the panel's (date, ts_ms)
     range (same warmup/mid_am/post_lunch/late quantile anchors and fuzz
     extension as ``mask_test.choose_truncation_points``);
  3. truncate the panel to the causal prefix of each cutoff
     (``date < T.date  OR  (date == T.date AND ts_ms <= T.ts_ms)``);
  4. recompute the prototype on the truncated panel;
  5. assert EXACT equality of the two columns over the prefix scope.

Directional semantics follow ``mask_test.compare_prefix``: rows present in
the truncated run must equal the full run's value, and rows of the full
prefix missing from the truncated output are flagged.  One simplification:
prototypes never recompute label columns, so the label exemption of the
engine test (compare labels only for ``ts <= T - H_max``) degenerates -- the
comparison scope for the prototype column is the full prefix ``<= T``.

A prototype that reads future rows (negative shifts, whole-series moments,
sorting by future values...) produces values near the cutoff that change
when the future is removed, and is REJECTED here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import polars as pl

from .registry import Prototype

__all__ = [
    "PanelCutoff",
    "PanelPrefixDiff",
    "PanelCausalityReport",
    "choose_panel_cutoffs",
    "compare_panel_prefix",
    "panel_prefix_check",
]

#: cutoff labels in canonical order with their (date, ts_ms) quantile targets;
#: mirrors validation/mask_test._BASE_POINTS (same session-phase anchors).
_BASE_POINTS: tuple[tuple[str, float], ...] = (
    ("warmup", 0.05),
    ("mid_am", 0.35),
    ("post_lunch", 0.60),
    ("late", 0.90),
)

_KEY_COLS = ("date", "instrument", "ts_ms")


@dataclass(frozen=True)
class PanelCutoff:
    """A global-time cut: keep rows with (date, ts_ms) <= (date, ts_ms)."""

    label: str
    date: str
    ts_ms: int

    def scope_expr(self) -> pl.Expr:
        """Polars predicate selecting the causal prefix of this cutoff."""
        return (pl.col("date") < self.date) | (
            (pl.col("date") == self.date) & (pl.col("ts_ms") <= self.ts_ms)
        )


@dataclass
class PanelPrefixDiff:
    identical: bool
    n_rows_scope: int
    first_diff: str | None = None


@dataclass
class PanelCausalityReport:
    prototype: str
    points: list[PanelCutoff] = field(default_factory=list)
    diffs: list[PanelPrefixDiff] = field(default_factory=list)
    passed: bool = False


def choose_panel_cutoffs(
    panel: pl.DataFrame, k: int = 4, seed: int = 42
) -> list[PanelCutoff]:
    """Pick K cutoffs spread over the panel's global (date, ts_ms) range.

    Deterministic for a given (panel, k, seed).  Cutoffs are quantiles of
    the sorted unique (date, ts_ms) pairs, so multi-day panels get cuts at
    real session phases (ts_ms is time-of-day and repeats across days).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    pairs = (
        panel.select(["date", "ts_ms"])
        .unique()
        .sort(["date", "ts_ms"])
    )
    n = pairs.height
    if n == 0:
        raise ValueError("panel has no (date, ts_ms) rows to cut on")
    dates = pairs["date"].to_list()
    tss = pairs["ts_ms"].to_list()

    def pick(frac: float) -> PanelCutoff:
        i = min(n - 1, max(0, int(round(frac * (n - 1)))))
        return PanelCutoff(label="", date=str(dates[i]), ts_ms=int(tss[i]))

    if k <= len(_BASE_POINTS):
        # keep warmup/late anchored; choose an evenly spread subset otherwise
        if k == 1:
            chosen = [_BASE_POINTS[3]]
        elif k == 2:
            chosen = [_BASE_POINTS[0], _BASE_POINTS[3]]
        elif k == 3:
            chosen = [_BASE_POINTS[0], _BASE_POINTS[1], _BASE_POINTS[3]]
        else:
            chosen = list(_BASE_POINTS)
    else:
        chosen = list(_BASE_POINTS)
        rng = random.Random(seed)
        i = 0
        seen = {label for label, _ in chosen}
        while len(chosen) < k:
            label = f"fuzz_{i}"
            frac = rng.uniform(0.05, 0.95)
            if label not in seen:
                chosen.append((label, frac))
                seen.add(label)
            i += 1

    points = []
    for label, frac in chosen:
        p = pick(frac)
        points.append(PanelCutoff(label=label, date=p.date, ts_ms=p.ts_ms))
    points.sort(key=lambda p: (p.date, p.ts_ms))
    return points


def _scope_key_arrays(df: pl.DataFrame, cutoff: PanelCutoff) -> list[tuple]:
    scoped = df.filter(cutoff.scope_expr()).sort(list(_KEY_COLS))
    return list(
        zip(
            scoped["date"].to_list(),
            scoped["instrument"].to_list(),
            scoped["ts_ms"].to_list(),
        )
    )


def compare_panel_prefix(
    full_panel: pl.DataFrame,
    trunc_panel: pl.DataFrame,
    column: str,
    cutoff: PanelCutoff,
) -> PanelPrefixDiff:
    """Compare one column between full and truncated runs over the prefix.

    Both frames must already carry the computed ``column``.  Rows are keyed
    by (date, instrument, ts_ms); the comparison scope is the causal prefix
    of ``cutoff``.  Values are compared EXACTLY, with null == null and
    NaN == NaN (warm-up must line up too).  Directional semantics mirror
    ``mask_test.compare_prefix``: truncated-present rows must equal the full
    run; full-prefix rows missing from the truncated output are flagged.
    """
    for df, tag in ((full_panel, "full"), (trunc_panel, "truncated")):
        for col in (*_KEY_COLS, column):
            if col not in df.columns:
                raise KeyError(f"{tag} panel lacks column {col!r}")

    f_scoped = full_panel.filter(cutoff.scope_expr()).sort(list(_KEY_COLS))
    t_scoped = trunc_panel.filter(cutoff.scope_expr()).sort(list(_KEY_COLS))

    f_keys = _scope_key_arrays(full_panel, cutoff)
    t_keys = _scope_key_arrays(trunc_panel, cutoff)

    n_scope = len(f_keys)
    first_diff: str | None = None

    t_set = set(t_keys)
    extra = [k for k in t_keys if k not in set(f_keys)]
    if extra:
        d, inst, ts = extra[0]
        first_diff = (
            f"row in truncated panel missing from full prefix: "
            f"date={d} instrument={inst} ts_ms={ts}"
        )
    if first_diff is None:
        f_set = set(f_keys)
        missing = [k for k in f_keys if k not in t_set]
        if missing:
            d, inst, ts = missing[0]
            first_diff = (
                f"row in full prefix missing from truncated panel: "
                f"date={d} instrument={inst} ts_ms={ts}"
            )

    if first_diff is None and f_keys != t_keys:
        first_diff = "row key order/identity mismatch between full and truncated scope"

    if first_diff is None:
        a = f_scoped[column].to_numpy().astype(np.float64)
        b = t_scoped[column].to_numpy().astype(np.float64)
        # polars maps null -> NaN in to_numpy, so equal_nan covers null==null
        if not np.array_equal(a, b, equal_nan=True):
            bad = np.flatnonzero(~((a == b) | (np.isnan(a) & np.isnan(b))))
            i = int(bad[0])
            d, inst, ts = f_keys[i]
            first_diff = (
                f"value mismatch at date={d} instrument={inst} ts_ms={ts} "
                f"col={column}: full={a[i]!r} trunc={b[i]!r}"
            )

    return PanelPrefixDiff(
        identical=first_diff is None, n_rows_scope=n_scope, first_diff=first_diff
    )


def panel_prefix_check(
    panel: pl.DataFrame,
    proto: Prototype,
    *,
    k: int = 4,
    seed: int = 42,
    compute_column: Callable[[pl.DataFrame, Prototype], pl.DataFrame] | None = None,
) -> PanelCausalityReport:
    """Run the truncate-and-recompute prefix identity test for a prototype.

    ``compute_column`` defaults to ``runner.compute_prototype_column`` (the
    indirection keeps this module import-cycle-free and lets tests inject
    instrumented computes).  The prototype is computed once on the full
    panel and once per truncated prefix; any value mismatch rejects it.
    """
    if compute_column is None:
        from .runner import compute_prototype_column as compute_column  # noqa: WPS433

    if panel.is_empty():
        raise ValueError("panel_prefix_check: panel is empty")

    full_aug = compute_column(panel, proto)
    points = choose_panel_cutoffs(panel, k=k, seed=seed)

    diffs: list[PanelPrefixDiff] = []
    for cutoff in points:
        trunc = panel.filter(cutoff.scope_expr())
        trunc_aug = compute_column(trunc, proto)
        diffs.append(
            compare_panel_prefix(full_aug, trunc_aug, proto.name, cutoff)
        )

    return PanelCausalityReport(
        prototype=proto.name,
        points=points,
        diffs=diffs,
        passed=all(d.identical for d in diffs),
    )
