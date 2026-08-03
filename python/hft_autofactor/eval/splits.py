"""Day-blocked purged walk-forward splits and the IS/OOS retention gate.

Labels are strictly intraday (the LabelBuilder emits ABSENT across lunch /
close / session end), so blocking by trading day SELF-PURGES the 900s
horizon: no training label window can reach into a test day.  An explicit
embargo of >= 1 trading day is still removed between train end and test
start to absorb any residual serial dependence (per docs/knowledge/
lookahead_prevention.md), and never tuned after the fact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Split:
    train_dates: tuple[str, ...]
    test_dates: tuple[str, ...]


def purged_day_splits(
    dates: Sequence[str],
    *,
    n_test_days: int = 5,
    mode: str = "anchored",  # "anchored" | "rolling"
    rolling_train_days: int | None = None,
    embargo_days: int = 1,
) -> list[Split]:
    """Walk-forward folds over sorted trading days.

    Test blocks are consecutive non-overlapping windows of ``n_test_days``
    days.  For a test block starting at index ``s``, training uses days
    ``[: s - embargo_days]`` (anchored) or the last ``rolling_train_days``
    of those (rolling).  Folds whose embargo leaves no training days are
    skipped -- the first test block therefore never starts at day 0.
    """
    if mode not in ("anchored", "rolling"):
        raise ValueError(f"unknown mode {mode!r} (use 'anchored' or 'rolling')")
    if mode == "rolling" and (rolling_train_days is None or rolling_train_days < 1):
        raise ValueError("rolling mode requires rolling_train_days >= 1")
    if n_test_days < 1:
        raise ValueError("n_test_days must be >= 1")
    if embargo_days < 0:
        raise ValueError("embargo_days must be >= 0")

    ordered = sorted(set(dates))
    n = len(ordered)
    splits: list[Split] = []
    for s in range(0, n, n_test_days):
        test = ordered[s : s + n_test_days]
        train_end = s - embargo_days
        if train_end <= 0 or not test:
            continue
        if mode == "anchored":
            train = ordered[:train_end]
        else:
            train = ordered[max(0, train_end - rolling_train_days) : train_end]
        if not train:
            continue
        splits.append(Split(train_dates=tuple(train), test_dates=tuple(test)))
    return splits


def is_oos_retention(
    is_ic: float, oos_ic: float, *, min_retention: float = 0.5
) -> bool:
    """OOS/IS IC retention check (McLean-Pontiff decay budget).

    Requires same sign and ``|oos_ic| >= min_retention * |is_ic|``.  Factors
    with negative IC are legitimately admissible inverted, so retention is
    defined on magnitudes; the sign check rejects flips.
    """
    if not (math.isfinite(is_ic) and math.isfinite(oos_ic)):
        return False
    if is_ic == 0.0:
        return False
    if math.copysign(1.0, is_ic) != math.copysign(1.0, oos_ic):
        return False
    return abs(oos_ic) >= min_retention * abs(is_ic)
