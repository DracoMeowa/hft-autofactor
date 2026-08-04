"""Pairwise factor correlation and greedy redundancy clustering.

Correlations are Spearman on pairwise-complete rows (re-using the same
rank-correlation primitive as the eval stage) computed on a stride-sampled
slice of the panel -- the full 3-second panel is far too large to rank in
memory, and a spread-out sample preserves the cross-factor dependence
structure while keeping the cost bounded.

Clusters are formed by greedy single-linkage union-find: factor pairs are
visited in descending |corr| order and unioned while |corr| >= threshold
(default 0.70).  Clusters of size >= 2 are "redundant families": they carry
overlapping information, so the proposer should mutate across families
rather than within them.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from ..eval.ic import spearman

__all__ = ["pairwise_spearman", "greedy_clusters", "cluster_name"]

DEFAULT_CORR_THRESHOLD = 0.70

#: primary information dimension per library factor (see coverage.py for the
#: full many-to-many map); used only to NAME clusters.
_FAMILY_OF = {
    "quoted_spread_ticks": "quote",
    "microprice_dev": "quote",
    "oir": "depth",
    "wdi": "depth",
    "book_slope": "depth",
    "iopv_premium": "iopv",
    "rv_60s": "quote",
    "rv_300s": "quote",
    "ofi_60s": "flow",
    "trade_imbalance_60s": "flow",
    "order_arrival_60s": "flow",
    "cancel_ratio_60s": "flow",
}

_FAMILY_ZH = {
    "depth": "深度族",
    "quote": "报价族",
    "flow": "订单流族",
    "iopv": "IOPV 族",
    "time_of_day": "日内时段族",
}


def pairwise_spearman(
    panel: pl.DataFrame, factors: Sequence[str]
) -> dict[tuple[str, str], float]:
    """Spearman correlation for every unordered factor pair.

    Pairwise-complete semantics (rows where either side is NaN/null are
    skipped for that pair only).  Pairs without enough overlap are NaN.
    """
    cols: dict[str, np.ndarray] = {}
    for f in factors:
        if f not in panel.columns:
            raise KeyError(f"panel lacks factor column {f!r}")
        cols[f] = panel[f].to_numpy().astype(np.float64)

    out: dict[tuple[str, str], float] = {}
    flist = list(factors)
    for i in range(len(flist)):
        for j in range(i + 1, len(flist)):
            a, b = flist[i], flist[j]
            out[(a, b)] = spearman(cols[a], cols[b])
    return out


def _union_find(factors: Sequence[str]):
    """Tiny union-find: returns (find, union) closures."""
    parent = {f: f for f in factors}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return find, union


def greedy_clusters(
    corr: Mapping[tuple[str, str], float],
    *,
    threshold: float = DEFAULT_CORR_THRESHOLD,
) -> list[dict]:
    """Greedy single-linkage clusters over |corr| >= threshold.

    Returns clusters of size >= 2, each::

        {"members": [...], "mean_abs_corr": float,
         "edges": [[a, b, corr], ...], "name": str}
    """
    factors: list[str] = []
    for a, b in corr:
        for f in (a, b):
            if f not in factors:
                factors.append(f)

    edges = [
        (a, b, float(v))
        for (a, b), v in corr.items()
        if v is not None and math.isfinite(v) and abs(v) >= threshold
    ]
    edges.sort(key=lambda e: -abs(e[2]))

    find, union = _union_find(factors)
    for a, b, _ in edges:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for f in factors:
        groups.setdefault(find(f), []).append(f)

    out: list[dict] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        member_edges = [
            [a, b, v] for a, b, v in edges
            if a in member_set and b in member_set
        ]
        # mean |corr| over ALL within-cluster pairs (not just linking edges)
        pair_vals = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                v = corr.get((members[i], members[j]))
                if v is None:
                    v = corr.get((members[j], members[i]))
                if v is not None and math.isfinite(v):
                    pair_vals.append(abs(v))
        out.append(
            {
                "members": sorted(members),
                "mean_abs_corr": (
                    float(np.mean(pair_vals)) if pair_vals else float("nan")
                ),
                "edges": member_edges,
                "name": cluster_name(sorted(members)),
            }
        )
    out.sort(key=lambda c: (-len(c["members"]), -c["mean_abs_corr"]))
    return out


def cluster_name(members: Sequence[str]) -> str:
    """Human-readable family name: shared dimension, else 混合族."""
    fams = {_FAMILY_OF.get(m) for m in members}
    fams.discard(None)
    if len(fams) == 1:
        fam = next(iter(fams))
        return f"{_FAMILY_ZH.get(fam, fam)}（{fam}）"
    if not fams:
        return "未映射族"
    return "混合族"
