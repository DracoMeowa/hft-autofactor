"""Signal-to-position mapping for short-horizon factors.

Causality contract
------------------
Everything in this module is strictly causal: the z-score at row ``t`` uses
only factor values at rows ``<= t`` (a trailing window), and the position at
row ``t`` is derived from z-scores at rows ``<= t - signal_lag_rows``.  The
mandatory ``signal_lag_rows >= 1`` is the tradability margin: a decision made
on the snapshot at time ``t`` can only be actuated at ``t + lag``.

Position rule
-------------
``position_from_z`` implements a hysteresis (entry/exit band) rule:

* enter a position of ``direction * max_position_units`` when ``|z|`` first
  reaches ``entry_z`` (sign of ``z * direction`` chooses long vs short);
* hold it while ``exit_z <= |z| < entry_z`` (or z is NaN: no signal, keep the
  established state);
* exit to 0 when ``|z|`` falls below ``exit_z``;
* reverse directly to the opposite side when ``|z| >= entry_z`` with the
  opposite sign;
* rows where ``tradable`` is False force the position to 0 and reset the
  state (a fresh entry signal is required afterwards);
* positions are shifted by ``signal_lag_rows`` (the actuation lag).

The engine (backtest/engine.py) clips negative targets to 0 -- there is no
spot shorting of A-share ETFs in v1, so a negative target means "flat".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl

__all__ = [
    "causal_zscore",
    "zscore_column",
    "PositionRule",
    "position_from_z",
]


def causal_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling z-score of ``values`` over a trailing ``window``.

    Output ``i`` uses only ``values[i-window+1 .. i]``; the first
    ``window - 1`` entries are NaN (warm-up, never zero-filled).  NaN inputs
    inside a window are ignored (z computed on the valid members); windows
    with fewer than 2 valid values yield NaN.  A constant window (zero std)
    yields 0.0 (neutral -- no entry signal).
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if n < window:
        return out

    wins = np.lib.stride_tricks.sliding_window_view(values, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        count = np.isfinite(wins).sum(axis=1)
        mean = np.nanmean(wins, axis=1)
        std = np.nanstd(wins, axis=1)

    z = np.full(n - window + 1, np.nan, dtype=np.float64)
    valid = count >= 2
    constant = valid & (std == 0.0)
    spread = valid & (std > 0.0)
    z[constant] = 0.0
    z[spread] = (values[window - 1:][spread] - mean[spread]) / std[spread]
    out[window - 1:] = z
    return out


def zscore_column(
    panel: "pl.DataFrame",
    factor: str,
    *,
    window_rows: int = 100,
    by: tuple[str, ...] = ("instrument", "date"),
) -> "pl.DataFrame":
    """Append a causal rolling z-score column ``f"{factor}_z"`` to ``panel``.

    The z-score is computed independently within each ``by`` group (default:
    per instrument and per day -- no cross-day state is ever carried, matching
    the engine contract that the channel/instrument mapping changes daily).
    Row order of the input frame is preserved.
    """
    import polars as pl

    out_name = f"{factor}_z"
    if panel.height == 0:
        return panel.with_columns(pl.Series(out_name, [], dtype=pl.Float64))
    for col in by:
        if col not in panel.columns:
            raise ValueError(f"zscore_column: grouping column {col!r} not in panel")
    if factor not in panel.columns:
        raise ValueError(f"zscore_column: factor column {factor!r} not in panel")

    indexed = panel.with_row_index("__hftaf_row")
    parts = []
    for _key, g in indexed.group_by(list(by), maintain_order=True):
        z = causal_zscore(g[factor].to_numpy(), window_rows)
        parts.append(
            pl.DataFrame(
                {
                    "__hftaf_row": g["__hftaf_row"].to_numpy(),
                    out_name: z,
                }
            )
        )
    zdf = pl.concat(parts).sort("__hftaf_row").drop("__hftaf_row")
    # zdf is aligned row-for-row with `indexed` (both back in original order)
    return indexed.drop("__hftaf_row").with_columns(zdf[out_name])


@dataclass
class PositionRule:
    """Hysteresis position rule applied to a causal z-score signal."""

    entry_z: float = 2.0
    exit_z: float = 0.5
    direction: int = 1  # +1: high factor => long; -1: inverted
    max_position_units: int = 100_000  # multiple of lot
    signal_lag_rows: int = 1  # actuation lag, tradability margin (>= 1)


def position_from_z(
    z: np.ndarray,
    rule: PositionRule,
    tradable: np.ndarray,
) -> np.ndarray:
    """Target position in fund units per row from a z-score signal.

    Returns a float array of signed targets: ``0`` (flat) or
    ``±rule.max_position_units``.  Rows where ``tradable`` is False are forced
    to 0 and reset the hysteresis state.  Decisions use z shifted back by
    ``rule.signal_lag_rows`` rows (tradability margin), so the position at
    row ``t`` depends only on ``z[t - lag]`` and earlier -- never the future.

    Note: A-share ETF spot markets have no shorting; downstream engines clip
    negative targets to 0 (``direction=-1`` therefore trades the inverted
    signal long/flat).
    """
    z = np.asarray(z, dtype=np.float64)
    tradable = np.asarray(tradable, dtype=bool)
    if z.ndim != 1 or tradable.ndim != 1:
        raise ValueError("z and tradable must be 1-D arrays")
    if z.shape[0] != tradable.shape[0]:
        raise ValueError("z and tradable must have the same length")
    if rule.entry_z <= 0:
        raise ValueError(f"entry_z must be > 0, got {rule.entry_z}")
    if not (0 <= rule.exit_z < rule.entry_z):
        raise ValueError(
            f"require 0 <= exit_z < entry_z, got exit_z={rule.exit_z}, "
            f"entry_z={rule.entry_z}"
        )
    if rule.direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1, got {rule.direction}")
    if rule.max_position_units < 0:
        raise ValueError("max_position_units must be >= 0")
    if rule.signal_lag_rows < 1:
        raise ValueError(
            f"signal_lag_rows must be >= 1 (tradability margin), "
            f"got {rule.signal_lag_rows}"
        )

    n = z.shape[0]
    pos = np.zeros(n, dtype=np.float64)
    lag = int(rule.signal_lag_rows)
    entry = float(rule.entry_z)
    exit_ = float(rule.exit_z)
    max_units = float(rule.max_position_units)
    direction = int(rule.direction)

    state = 0  # 0 flat, +1 long, -1 short (pre-clip target sign)
    for i in range(n):
        if not bool(tradable[i]):
            # Untradable row: position forced flat; a fresh entry signal is
            # required once the market becomes tradable again.
            state = 0
            pos[i] = 0.0
            continue
        j = i - lag
        zi = float(z[j]) if j >= 0 else float("nan")
        if np.isfinite(zi):
            sign = 1 if zi > 0 else (-1 if zi < 0 else 0)
            azi = abs(zi)
        else:
            sign = 0
            azi = float("nan")

        if state == 0:
            if np.isfinite(zi) and azi >= entry and sign != 0:
                state = sign * direction
        else:
            if np.isfinite(zi) and azi < exit_:
                state = 0
            elif np.isfinite(zi) and azi >= entry and sign != 0:
                wanted = sign * direction
                if wanted != state:
                    state = wanted  # direct reversal at the entry band
            # NaN z (signal dropout): hold the established state
        pos[i] = state * max_units
    return pos
