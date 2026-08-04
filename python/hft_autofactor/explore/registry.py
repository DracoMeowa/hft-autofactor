"""Prototype registry: metadata-complete registration of factor prototypes.

A prototype is a 5-tuple ``(name, economic mechanism, info set, inspiration,
compute spec)``.  Registration REQUIRES every metadata field to be present
and non-empty -- incomplete entries are refused with a :class:`PrototypeError`
listing exactly what is missing.  No silent defaults: an undocumented
prototype is an un-auditable one.

Compute-spec contract
---------------------
``compute(part) -> pl.Series | np.ndarray | pl.DataFrame`` where ``part`` is
ONE ``(date, instrument)`` group of the panel, sorted by ``ts_ms`` ascending,
containing the base columns + library factor columns but NEVER the label
columns (``fwd_*_ret_*``): the runner strips labels before the call, so a
prototype's info set cannot include the prediction targets.  The return value
must align row-for-row with ``part`` (same length); warm-up rows are null,
never zero-filled.  Only backward-looking windows are legal (per-instrument,
per-day grouping, no future rows) -- the panel prefix causality test
(:mod:`hft_autofactor.explore.causality`) enforces this empirically.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import polars as pl

from ..ingest import BASE_COLUMNS, DEFAULT_FACTORS

__all__ = [
    "Prototype",
    "PrototypeError",
    "PrototypeRegistry",
    "explore_prototype",
    "default_registry",
    "load_prototype_spec",
]

#: metadata fields that must be present and non-empty for registration
REQUIRED_METADATA: tuple[str, ...] = ("name", "mechanism", "info_set", "inspiration")

#: valid prototype names double as panel column names
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: column names a prototype may never take (would shadow panel columns)
_RESERVED_COLUMNS: frozenset[str] = frozenset(BASE_COLUMNS) | frozenset(
    DEFAULT_FACTORS
) | {"channel"}


class PrototypeError(ValueError):
    """A prototype was refused (incomplete metadata, bad name, collision)."""


@dataclass(frozen=True)
class Prototype:
    """One registered factor prototype.

    Attributes:
        name:        new panel column name (lower snake_case).
        mechanism:   the economic mechanism in plain language (WHY it should
                     predict short-horizon returns).
        info_set:    panel columns the compute spec reads.
        inspiration: source of the idea (paper, observation, prior factor).
        compute:     causal compute spec; see module docstring.
        source:      "builtin" or the spec file it was loaded from.
    """

    name: str
    mechanism: str
    info_set: str
    inspiration: str
    compute: Callable[[pl.DataFrame], object]
    source: str = "builtin"

    def metadata_dict(self) -> dict:
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "info_set": self.info_set,
            "inspiration": self.inspiration,
            "source": self.source,
        }


def _validate_fields(
    name: object,
    mechanism: object,
    info_set: object,
    inspiration: object,
    compute: object,
) -> None:
    """Refuse incomplete or malformed entries (the registration gate)."""
    missing = []
    values = {
        "name": name,
        "mechanism": mechanism,
        "info_set": info_set,
        "inspiration": inspiration,
    }
    for key in REQUIRED_METADATA:
        v = values[key]
        if v is None or not str(v).strip():
            missing.append(key)
    if compute is None or not callable(compute):
        missing.append("compute (callable)")
    if missing:
        raise PrototypeError(
            "prototype registration refused: missing/empty field(s): "
            + ", ".join(missing)
        )

    name_s = str(name).strip()
    if not _NAME_RE.match(name_s):
        raise PrototypeError(
            f"prototype name {name_s!r} invalid: want lower snake_case "
            "starting with a letter (it becomes a panel column name)"
        )
    if name_s in _RESERVED_COLUMNS or name_s.startswith("fwd_"):
        raise PrototypeError(
            f"prototype name {name_s!r} is reserved (shadows a panel column)"
        )


def explore_prototype(
    *,
    name: str,
    mechanism: str,
    info_set: str,
    inspiration: str,
    compute: Callable[[pl.DataFrame], object],
    source: str = "builtin",
) -> Prototype:
    """Build a :class:`Prototype`, refusing incomplete metadata."""
    _validate_fields(name, mechanism, info_set, inspiration, compute)
    return Prototype(
        name=str(name).strip(),
        mechanism=str(mechanism).strip(),
        info_set=str(info_set).strip(),
        inspiration=str(inspiration).strip(),
        compute=compute,
        source=source,
    )


class PrototypeRegistry:
    """A name-keyed collection of prototypes with collision checks."""

    def __init__(self, prototypes: Iterable[Prototype] = ()):
        self._protos: dict[str, Prototype] = {}
        for p in prototypes:
            self.register(p)

    def register(self, proto: Prototype, *, overwrite: bool = False) -> Prototype:
        """Add ``proto``; refuse duplicates unless ``overwrite``."""
        if not isinstance(proto, Prototype):
            raise PrototypeError(
                f"expected a Prototype, got {type(proto).__name__}"
            )
        _validate_fields(
            proto.name, proto.mechanism, proto.info_set,
            proto.inspiration, proto.compute,
        )
        if proto.name in self._protos and not overwrite:
            raise PrototypeError(
                f"prototype {proto.name!r} already registered "
                f"(source: {self._protos[proto.name].source})"
            )
        self._protos[proto.name] = proto
        return proto

    def get(self, name: str) -> Prototype:
        try:
            return self._protos[name]
        except KeyError:
            raise PrototypeError(
                f"unknown prototype {name!r}; registered: {', '.join(self.names())}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._protos)

    def __contains__(self, name: str) -> bool:
        return name in self._protos

    def __iter__(self):
        return iter(self._protos[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._protos)


# --------------------------------------------------------------------- #
# built-in seed prototypes                                              #
# --------------------------------------------------------------------- #
def _compute_log_mid_ret_60s(part: pl.DataFrame) -> pl.Series:
    """Trailing 60s (20 x 3s rows) log-mid return; warm-up rows null."""
    return part.select(
        pl.col("mid_px").log().diff(20).alias("value")
    )["value"]


def _compute_spread_z_300s(part: pl.DataFrame) -> pl.Series:
    """Causal z-score of the library quoted spread over 100 rows (300s).

    Constant trailing windows map to 0.0 (neutral), matching the
    backtest.causal_zscore convention; warm-up (< 100 rows) is null.
    """
    x = pl.col("quoted_spread_ticks")
    mean = x.rolling_mean(window_size=100, min_samples=100)
    std = x.rolling_std(window_size=100, min_samples=100)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


def _compute_depth_imbalance_5l(part: pl.DataFrame) -> pl.Series:
    """5-level depth imbalance (bid-ask)/(bid+ask); null when no depth."""
    b = pl.col("depth_bid5").cast(pl.Float64)
    a = pl.col("depth_ask5").cast(pl.Float64)
    return part.select(
        pl.when((b + a) > 0.0)
        .then((b - a) / (b + a))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


def default_registry() -> PrototypeRegistry:
    """The built-in seed prototypes (all strictly causal)."""
    return PrototypeRegistry(
        [
            explore_prototype(
                name="log_mid_ret_60s",
                mechanism=(
                    "signed 60-second mid-price momentum: tests whether ETF mids "
                    "drift (momentum) or snap back (mean reversion) after short "
                    "moves; the library rv_60s keeps only the magnitude."
                ),
                info_set="mid_px",
                inspiration=(
                    "short-horizon return autocorrelation (Lo & MacKinlay 1990); "
                    "signed companion of the engine's rv_60s."
                ),
                compute=_compute_log_mid_ret_60s,
            ),
            explore_prototype(
                name="spread_z_300s",
                mechanism=(
                    "liquidity-state signal: a quoted spread unusually wide vs "
                    "its own trailing 300s distribution flags stressed quoting "
                    "and wider adverse selection."
                ),
                info_set="quoted_spread_ticks (library)",
                inspiration=(
                    "spread persistence and state-dependence (Stoll 2003); "
                    "z-scored analogue of the library quoted_spread_ticks."
                ),
                compute=_compute_spread_z_300s,
            ),
            explore_prototype(
                name="depth_imbalance_5l",
                mechanism=(
                    "order-book pressure from the full 5-level depth: bid-heavy "
                    "books precede upward moves; the 5-level version of the "
                    "top-of-book oir."
                ),
                info_set="depth_bid5, depth_ask5",
                inspiration=(
                    "Cont, Stoikov & Talreja (2010) queue-reactive model; "
                    "deeper-depth extension of the library oir."
                ),
                compute=_compute_depth_imbalance_5l,
            ),
        ]
    )


# --------------------------------------------------------------------- #
# spec-file loading (CLI `add` and persisted registry)                  #
# --------------------------------------------------------------------- #
def load_prototype_spec(path: str | Path, *, source: str | None = None) -> Prototype:
    """Import a prototype spec file and return its validated Prototype.

    The module must define ``PROTOTYPE`` -- either the result of
    :func:`explore_prototype` or a plain dict with the same keys.  All
    metadata completeness rules apply; incomplete specs are refused.
    """
    path = Path(path)
    if not path.is_file():
        raise PrototypeError(f"spec file not found: {path}")
    module_name = f"hftaf_explore_spec_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PrototypeError(f"cannot import spec file: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PrototypeError(f"spec file {path} failed to import: {exc}") from exc

    proto = getattr(module, "PROTOTYPE", None)
    if proto is None:
        raise PrototypeError(
            f"spec file {path} must define PROTOTYPE (explore_prototype(...) "
            "result or dict with name/mechanism/info_set/inspiration/compute)"
        )
    if isinstance(proto, dict):
        proto = explore_prototype(**proto)
    if not isinstance(proto, Prototype):
        raise PrototypeError(
            f"spec file {path}: PROTOTYPE has type {type(proto).__name__}, "
            "want explore_prototype(...) or dict"
        )
    if source is None:
        source = str(path)
    if source != proto.source:
        proto = Prototype(
            name=proto.name,
            mechanism=proto.mechanism,
            info_set=proto.info_set,
            inspiration=proto.inspiration,
            compute=proto.compute,
            source=source,
        )
    return proto
