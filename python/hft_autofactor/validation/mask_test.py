"""Lookahead mask validation (Stage 2): truncate-and-recompute prefix identity.

Protocol (docs/validation_plan.md, MASK TEST A):
  1. run the deterministic engine on the FULL day streams -> full output CSV;
  2. choose K truncation points T spanning warmup / mid-morning / post-lunch /
     late session;
  3. truncate the tick file by SEQUENCE POSITION (SeqNo <= seq(T), where
     seq(T) is the max SeqNo among ticks with TransactTime <= T) and the
     snapshot file by UpdateTime <= T;
  4. rerun the SAME binary on the truncated streams;
  5. assert the truncated output equals the full output restricted to
     t <= T for factor/state/flag columns, and to t <= T - H_max for label
     columns (labels legitimately differ closer to the cut because they look
     H_max into the future).  One asymmetry is allowed even inside the label
     horizon: a label ABSENT in the truncated run but present in the full run
     is legitimate, because a label resolves at the FIRST snapshot U >= t+H,
     and U may fall after the cut when the instrument has snapshot gaps
     (sparse LOF/ETF books).  The converse is enforced: a label present in
     the truncated run must equal the full run's value.

Canary check: rerunning with ``--canaries`` (deliberate look-ahead factors)
MUST fail the prefix identity test -- proving the validator detects leakage.
"""
from __future__ import annotations

import csv
import dataclasses
import gzip
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..config import PipelineConfig
from ..ingest import DayJob, discover_jobs

#: truncation-point labels in canonical order with their ts quantile targets
_BASE_POINTS: tuple[tuple[str, float], ...] = (
    ("warmup", 0.05),
    ("mid_am", 0.35),
    ("post_lunch", 0.60),
    ("late", 0.90),
)

_SEQ_COLS = ("SeqNo", "ApplSeqNum")
_TICK_TIME_COLS = ("TransactTime",)
_SNAP_TIME_COLS = ("UpdateTime", "DataTime")

_LABEL_COL_PREFIXES = ("fwd_mid_ret_", "fwd_last_ret_")


@dataclass(frozen=True)
class TruncationPoint:
    """A cut point: keep ticks with SeqNo <= tick_seq, snapshots ts <= ts_ms.

    ``tick_seq == -1`` is the sentinel for "not yet resolved against the tick
    stream" (as returned by :func:`choose_truncation_points`); callers with
    access to the tick file must resolve it via :func:`resolve_tick_seq`
    before truncating.
    """

    label: str  # "warmup" | "mid_am" | "post_lunch" | "late" | "fuzz_i"
    ts_ms: int
    tick_seq: int  # keep ticks with SeqNo <= this


@dataclass
class PrefixDiff:
    identical: bool
    n_rows_full: int
    n_rows_trunc: int
    first_diff: str | None = None


@dataclass
class MaskReport:
    date: str
    exchange: str
    channel: int
    points: list[TruncationPoint] = field(default_factory=list)
    diffs: list[PrefixDiff] = field(default_factory=list)
    canary_run: bool = False
    canary_failed_as_required: bool = False
    passed: bool = False


# --------------------------------------------------------------------- #
# time helpers (fast path for HHMMSSmmm, tolerant of HH:MM:SS[.mmm])    #
# --------------------------------------------------------------------- #
def parse_time_ms(value: str) -> int | None:
    """Parse HHMMSSmmm / HH:MM:SS[.mmm] into ms since midnight.

    Digit strings of up to 9 chars are right-justified HHMMSSmmm: real
    SSE/SZSE dumps write the value as an integer with leading zeros dropped
    ("91400650" = 09:14:00.650). Consistent with the C++ engine parser.
    """
    s = value.strip()
    if not s:
        return None
    try:
        if ":" in s:
            hh, mm, rest = s.split(":")
            return int(hh) * 3_600_000 + int(mm) * 60_000 + int(round(float(rest) * 1000))
        if not s.isdigit() or len(s) > 9:
            return None
        v = int(s)
        ms = v % 1000; v //= 1000
        ss = v % 100;  v //= 100
        mm = v % 100;  v //= 100
        hh = v
        if hh > 23 or mm > 59 or ss > 59:
            return None
        return ((hh * 60 + mm) * 60 + ss) * 1000 + ms
    except (ValueError, IndexError):
        return None


def _column_index(header: Sequence[str], candidates: Sequence[str], what: str) -> int:
    lower = {c.strip().lower(): i for i, c in enumerate(header)}
    for cand in candidates:
        idx = lower.get(cand.lower())
        if idx is not None:
            return idx
    raise ValueError(f"column {what} not found; header={list(header)}")


# --------------------------------------------------------------------- #
# truncation point selection                                            #
# --------------------------------------------------------------------- #
def choose_truncation_points(
    full_csv: Path, k: int = 4, seed: int = 42
) -> list[TruncationPoint]:
    """Pick K cut times spread over the day's factor rows.

    Returns points sorted by ts_ms with ``tick_seq`` set to the unresolved
    sentinel -1 (resolve with :func:`resolve_tick_seq` against the tick
    file).  Deterministic for a given (full_csv, k, seed).
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    ts_values: list[int] = []
    with open(full_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        ts_idx = _column_index(header, ("ts_ms",), "ts_ms")
        for row in reader:
            if row:
                ts_values.append(int(row[ts_idx]))
    if not ts_values:
        raise ValueError(f"{full_csv}: no data rows to choose truncation points from")

    uniq = sorted(set(ts_values))
    n = len(uniq)

    def pick(frac: float) -> int:
        return uniq[min(n - 1, max(0, int(round(frac * (n - 1)))))]

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

    points = [TruncationPoint(label=label, ts_ms=pick(frac), tick_seq=-1)
              for label, frac in chosen]
    points.sort(key=lambda p: p.ts_ms)
    return points


def resolve_tick_seq(tick_gz: Path, ts_ms: int) -> int:
    """Max SeqNo among tick rows with TransactTime <= ``ts_ms`` (0 if none)."""
    return resolve_tick_seqs(tick_gz, [ts_ms])[0]


def resolve_tick_seqs(tick_gz: Path, ts_list: Sequence[int]) -> list[int]:
    """Resolve several cut times in ONE pass over the (large) tick stream."""
    targets = sorted(set(int(t) for t in ts_list))
    best: dict[int, int] = {t: 0 for t in targets}
    with gzip.open(tick_gz, "rt", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split(",")
        seq_idx = _column_index(header, _SEQ_COLS, "SeqNo/ApplSeqNum")
        time_idx = _column_index(header, _TICK_TIME_COLS, "TransactTime")
        for line in fh:
            fields = line.rstrip("\r\n").split(",")
            if len(fields) <= max(seq_idx, time_idx):
                continue
            t = parse_time_ms(fields[time_idx])
            if t is None:
                continue
            # targets ascending: once t exceeds a target it never matches again
            # only if time is monotone; do not assume -- check all remaining
            for target in targets:
                if t <= target:
                    seq = int(fields[seq_idx])
                    if seq > best[target]:
                        best[target] = seq
    return [best[int(t)] for t in ts_list]


# --------------------------------------------------------------------- #
# input truncation                                                      #
# --------------------------------------------------------------------- #
def truncate_tick_file(src_gz: Path, dst_gz: Path, max_seq: int) -> int:
    """Rewrite the tick gz keeping only rows with SeqNo <= ``max_seq``.

    Truncation by SEQUENCE POSITION (not timestamp) so ordering bugs surface.
    Returns the number of data rows kept.
    """
    kept = 0
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    # compresslevel=1: truncated inputs are transient; rewrite speed matters
    # far more than compression ratio (this loop dominates mask wall time).
    with gzip.open(src_gz, "rt", encoding="utf-8", newline="") as src, \
            gzip.open(dst_gz, "wt", encoding="utf-8", newline="\n",
                      compresslevel=1) as dst:
        header = src.readline().rstrip("\r\n")
        cols = header.split(",")
        seq_idx = _column_index(cols, _SEQ_COLS, "SeqNo/ApplSeqNum")
        dst.write(header + "\n")
        for line in src:
            fields = line.rstrip("\r\n").split(",")
            if len(fields) > seq_idx and int(fields[seq_idx]) <= max_seq:
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    return kept


def truncate_snapshot_file(src_gz: Path, dst_gz: Path, max_ts_ms: int) -> int:
    """Rewrite the snapshot gz keeping only rows with UpdateTime <= max_ts_ms."""
    kept = 0
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_gz, "rt", encoding="utf-8", newline="") as src, \
            gzip.open(dst_gz, "wt", encoding="utf-8", newline="\n",
                      compresslevel=1) as dst:
        header = src.readline().rstrip("\r\n")
        cols = header.split(",")
        time_idx = _column_index(cols, _SNAP_TIME_COLS, "UpdateTime")
        dst.write(header + "\n")
        for line in src:
            fields = line.rstrip("\r\n").split(",")
            if len(fields) <= time_idx:
                continue
            t = parse_time_ms(fields[time_idx])
            if t is not None and t <= max_ts_ms:
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1
    return kept


# --------------------------------------------------------------------- #
# prefix comparison                                                     #
# --------------------------------------------------------------------- #
def compare_prefix(
    full_csv: Path,
    trunc_csv: Path,
    cut_ts_ms: int,
    horizons_max_s: int = 900,
) -> PrefixDiff:
    """Assert truncated-run output equals the full-run prefix.

    Factor/state/flag columns are compared for ``ts_ms <= cut_ts_ms``;
    label columns (``fwd_*_ret_*``) only for ``ts_ms <= cut - horizons_max_s``
    because labels legitimately look into the future.  Within that label
    scope the check is directional: a label present in the truncated run
    must equal the full run's cell, but a label ABSENT in the truncated
    run is allowed -- resolution happens at the first snapshot U >= t+H,
    which may fall after the cut for instruments with snapshot gaps, so
    the full run can legitimately resolve labels the truncated run cannot.
    Comparison is exact string equality of CSV cells (legal under the
    bit-exact determinism contract for same-binary reruns).
    """
    def load(path: Path) -> tuple[list[str], dict[tuple[str, int], list[str]]]:
        rows: dict[tuple[str, int], list[str]] = {}
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            inst_idx = _column_index(header, ("instrument",), "instrument")
            ts_idx = _column_index(header, ("ts_ms",), "ts_ms")
            for row in reader:
                if not row:
                    continue
                key = (row[inst_idx], int(row[ts_idx]))
                rows[key] = row
        return header, rows

    full_header, full_rows = load(full_csv)
    trunc_header, trunc_rows = load(trunc_csv)

    if full_header != trunc_header:
        return PrefixDiff(False, len(full_rows), len(trunc_rows),
                          f"header mismatch: full={full_header} trunc={trunc_header}")

    inst_idx = full_header.index("instrument")
    ts_idx = full_header.index("ts_ms")
    label_cols = {
        i for i, c in enumerate(full_header)
        if any(c.startswith(p) for p in _LABEL_COL_PREFIXES)
    }
    label_cutoff = cut_ts_ms - horizons_max_s * 1000

    full_prefix = {k: r for k, r in full_rows.items() if k[1] <= cut_ts_ms}
    trunc_prefix = {k: r for k, r in trunc_rows.items() if k[1] <= cut_ts_ms}

    first_diff: str | None = None

    extra = sorted(k for k in trunc_prefix if k not in full_prefix)
    if extra:
        inst, ts = extra[0]
        first_diff = f"row in truncated output missing from full prefix: instrument={inst} ts_ms={ts}"

    if first_diff is None:
        for key in sorted(full_prefix):
            if key not in trunc_prefix:
                first_diff = (
                    f"row in full prefix missing from truncated output: "
                    f"instrument={key[0]} ts_ms={key[1]}"
                )
                break

    if first_diff is None:
        for key in sorted(full_prefix):
            f_row, t_row = full_prefix[key], trunc_prefix[key]
            if len(f_row) != len(t_row):
                first_diff = (
                    f"field count mismatch at instrument={key[0]} ts_ms={key[1]}: "
                    f"{len(f_row)} vs {len(t_row)}"
                )
                break
            compare_labels = key[1] <= label_cutoff
            for i, (a, b) in enumerate(zip(f_row, t_row)):
                if i in (inst_idx, ts_idx):
                    continue
                if i in label_cols:
                    if not compare_labels:
                        continue
                    if b == "":
                        # ABSENT in the truncated run: legitimate even inside
                        # the label horizon -- the resolving snapshot (first
                        # U >= t+H) may fall after the cut for instruments
                        # with snapshot gaps.  full-present/trunc-absent is
                        # therefore not a mismatch; trunc-present cells must
                        # still equal the full run's (checked below).
                        continue
                if a != b:
                    first_diff = (
                        f"value mismatch at instrument={key[0]} ts_ms={key[1]} "
                        f"col={full_header[i]}: full={a!r} trunc={b!r}"
                    )
                    break
            if first_diff is not None:
                break

    return PrefixDiff(
        identical=first_diff is None,
        n_rows_full=len(full_prefix),
        n_rows_trunc=len(trunc_prefix),
        first_diff=first_diff,
    )


# --------------------------------------------------------------------- #
# engine invocation                                                     #
# --------------------------------------------------------------------- #
def run_engine(engine_bin: Path, args: Sequence[str]) -> subprocess.CompletedProcess:
    """Run the hftaf-engine binary with ``args`` and capture output."""
    return subprocess.run(
        [str(engine_bin), *[str(a) for a in args]],
        capture_output=True,
        text=True,
    )


def build_id_for(engine_bin: Path) -> str:
    """Cheap reproducible build identifier from the binary's stat."""
    try:
        st = Path(engine_bin).stat()
        return f"{st.st_mtime_ns:x}-{st.st_size:x}"
    except OSError:
        return "unknown"


def engine_cli_args(
    cfg: PipelineConfig,
    *,
    exchange: str,
    date: str,
    channel: int,
    tick_gz: Path,
    snapshot_gz: Path,
    out_csv: Path,
    canaries: bool = False,
    build_id: str | None = None,
) -> list[str]:
    """Build the canonical hftaf-engine CLI argument list."""
    args = [
        "--exchange", exchange,
        "--date", date,
        "--channel", str(channel),
        "--ticks", str(tick_gz),
        "--snapshots", str(snapshot_gz),
        "--out", str(out_csv),
        "--horizons", ",".join(str(h) for h in cfg.horizons_s),
    ]
    if cfg.factors:
        args += ["--factors", ",".join(cfg.factors)]
    if canaries:
        args.append("--canaries")
    args += ["--build-id", build_id or build_id_for(cfg.engine_bin)]
    return args


def _run_or_raise(engine_bin: Path, args: Sequence[str], what: str) -> None:
    cp = run_engine(engine_bin, args)
    if cp.returncode != 0:
        raise RuntimeError(
            f"engine failed ({what}); rc={cp.returncode}\n"
            f"args: {' '.join(str(a) for a in args)}\n"
            f"stderr tail: {cp.stderr[-2000:]}"
        )


# --------------------------------------------------------------------- #
# whole-day mask test                                                   #
# --------------------------------------------------------------------- #
def mask_test_day(
    cfg: PipelineConfig,
    date: str,
    exchange: str,
    channel: int,
    *,
    k: int = 4,
    include_canaries: bool = True,
) -> MaskReport:
    """Run the full truncate-and-recompute mask test for one job.

    Steps: ensure the full-run output exists (running the engine if needed),
    choose+resolve K truncation points, rerun the engine per truncated input
    pair and compare prefixes, then rerun with ``--canaries`` which MUST
    break prefix identity (otherwise the validator is blind to leakage).
    """
    jobs = [
        j
        for j in discover_jobs(cfg, [date])
        if j.exchange == exchange and j.channel == channel
    ]
    if not jobs:
        raise FileNotFoundError(
            f"no discoverable job for date={date} exchange={exchange} channel={channel}"
        )
    job: DayJob = jobs[0]
    cfg.ensure_dirs()

    work = cfg.validation_dir / "work" / f"{date}_{exchange}_ch{channel}"
    work.mkdir(parents=True, exist_ok=True)
    max_horizon = max(cfg.horizons_s) if cfg.horizons_s else 900

    full_out = job.out_csv
    if not full_out.exists():
        _run_or_raise(
            cfg.engine_bin,
            engine_cli_args(
                cfg,
                exchange=exchange, date=date, channel=channel,
                tick_gz=job.tick_gz, snapshot_gz=job.snapshot_gz,
                out_csv=full_out, canaries=False,
            ),
            what="full run",
        )

    points = choose_truncation_points(full_out, k=k)
    seqs = resolve_tick_seqs(job.tick_gz, [p.ts_ms for p in points])
    resolved = [
        dataclasses.replace(p, tick_seq=seq) for p, seq in zip(points, seqs)
    ]

    diffs: list[PrefixDiff] = []
    for i, p in enumerate(resolved):
        t_gz = work / f"ticks_cut{i}.csv.gz"
        s_gz = work / f"snaps_cut{i}.csv.gz"
        truncate_tick_file(job.tick_gz, t_gz, p.tick_seq)
        truncate_snapshot_file(job.snapshot_gz, s_gz, p.ts_ms)
        trunc_out = work / f"out_cut{i}.csv"
        _run_or_raise(
            cfg.engine_bin,
            engine_cli_args(
                cfg,
                exchange=exchange, date=date, channel=channel,
                tick_gz=t_gz, snapshot_gz=s_gz, out_csv=trunc_out,
                canaries=False,
            ),
            what=f"truncated run {p.label}",
        )
        diffs.append(compare_prefix(full_out, trunc_out, p.ts_ms, max_horizon))

    canary_run = False
    canary_failed_as_required = False
    if include_canaries:
        canary_full = work / "out_canary_full.csv"
        _run_or_raise(
            cfg.engine_bin,
            engine_cli_args(
                cfg,
                exchange=exchange, date=date, channel=channel,
                tick_gz=job.tick_gz, snapshot_gz=job.snapshot_gz,
                out_csv=canary_full, canaries=True,
            ),
            what="canary full run",
        )
        # use the latest cut point: canary rows in (T-H, T] must differ.
        # The truncated inputs for that point were already written by the
        # loop above (same tick_seq / ts_ms) -- reuse them instead of
        # rewriting hundreds of MB of gzip a second time.
        p = resolved[-1]
        t_gz = work / f"ticks_cut{len(resolved) - 1}.csv.gz"
        s_gz = work / f"snaps_cut{len(resolved) - 1}.csv.gz"
        canary_trunc = work / "out_canary_trunc.csv"
        _run_or_raise(
            cfg.engine_bin,
            engine_cli_args(
                cfg,
                exchange=exchange, date=date, channel=channel,
                tick_gz=t_gz, snapshot_gz=s_gz, out_csv=canary_trunc,
                canaries=True,
            ),
            what="canary truncated run",
        )
        canary_diff = compare_prefix(canary_full, canary_trunc, p.ts_ms, max_horizon)
        canary_run = True
        canary_failed_as_required = not canary_diff.identical

    passed = all(d.identical for d in diffs) and (
        (not include_canaries) or canary_failed_as_required
    )
    return MaskReport(
        date=date,
        exchange=exchange,
        channel=channel,
        points=resolved,
        diffs=diffs,
        canary_run=canary_run,
        canary_failed_as_required=canary_failed_as_required,
        passed=passed,
    )
