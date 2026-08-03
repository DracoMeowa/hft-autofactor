"""Golden hashes for engine outputs.

``hash_output_csv`` computes a sha256 over NORMALIZED rows of an interchange
CSV: rows are sorted by (instrument, ts_ms), each row is re-joined with a
canonical separator and LF endings, so the digest is stable against row order
but sensitive to any value change.  With the deterministic-build engine
(-O2 -fno-fast-math -ffp-contract=off, int64-scaled prices) a re-run on the
same binary/machine must reproduce the digest exactly; cross-machine
comparisons fall back to the ULP-tolerance mode documented in
docs/validation_plan.md instead of hash equality.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..config import PipelineConfig

_FIELD_SEP = ","
_ROW_SEP = "\n"


def _golden_path(cfg: PipelineConfig, date: str, exchange: str, channel: int) -> Path:
    return cfg.golden_dir / f"{date}__{exchange}__ch{channel}.sha256"


def hash_output_csv(csv_path: Path) -> str:
    """sha256 over normalized (sorted) rows of an interchange CSV."""
    header = ""
    rows: list[tuple[str, int, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n")
        cols = header.split(",")
        try:
            inst_idx = cols.index("instrument")
            ts_idx = cols.index("ts_ms")
        except ValueError as exc:
            raise ValueError(f"{csv_path}: not an interchange CSV (missing columns)") from exc
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split(",")
            rows.append((fields[inst_idx], int(fields[ts_idx]), line))

    rows.sort(key=lambda r: (r[0], r[1]))
    payload = header + _ROW_SEP + _ROW_SEP.join(r[2] for r in rows) + _ROW_SEP
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def store_golden(
    date: str, exchange: str, channel: int, digest: str, cfg: PipelineConfig
) -> None:
    """Persist a golden digest under ``validation_dir/golden``."""
    path = _golden_path(cfg, date, exchange, channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest + "\n", encoding="utf-8")


def load_golden(
    date: str, exchange: str, channel: int, cfg: PipelineConfig
) -> str | None:
    """Load a previously stored golden digest, or None if absent."""
    path = _golden_path(cfg, date, exchange, channel)
    if not path.is_file():
        return None
    digest = path.read_text(encoding="utf-8").strip()
    return digest or None
