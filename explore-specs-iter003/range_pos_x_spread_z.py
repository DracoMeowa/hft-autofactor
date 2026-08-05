"""Explore-lane prototype spec (iter-003 round 2, day-range/OHLC family R2-A).

range_pos_x_spread_z: champion interaction -- centered day-range position
times stressed-quoting state. Liquidity withdrawal AT the anchoring zones.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

EPS = 1e-12
W_SPREAD = 100  # 100 x 3s rows = 300s z window on the spread state


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(day_range_pos - 0.5) * z(quoted_spread_ticks, 300s)."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    cpos = pos - 0.5
    sp_z = _z(pl.col("quoted_spread_ticks"), W_SPREAD)
    return part.select((cpos * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_pos_x_spread_z",
    mechanism=(
        "Quoting stress located at the range zones: a spread unusually wide "
        "for its own recent history WHILE price sits near a day boundary "
        "flags market makers withdrawing exactly where resting stops "
        "cluster -- adverse-selection fear ahead of a boundary resolution. "
        "Wide-spread-near-high (positive interaction) warns the breakout "
        "zone is being treated as toxic: resolution, when it comes, tends "
        "to be sharp, and the side defending the level has information "
        "advantage; tight-spread-at-boundary is confident quoting and "
        "absorption. Bare spread level was IS-dead in round 1 while "
        "spread-z interactions passed 15s twice -- spread only carries "
        "signal as a CONDITION, and here the conditioning dimension is the "
        "range-position champion itself."
    ),
    info_set="mid_px, quoted_spread_ticks",
    inspiration=(
        "iter-003 round-2 R2-A family brief direction 6 (champion x spread "
        "state); direct transfer of the round-1 ofi_z_x_spread_z / "
        "flow_divergence_x_spread_z conditioning lesson to the anchoring "
        "dimension (Stoll 2003 on spread state-dependence)."
    ),
    compute=compute,
)
