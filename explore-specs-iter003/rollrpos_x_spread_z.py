"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

rollrpos_x_spread_z: the admitted range_pos_x_spread_z question moved to
the LOCAL rolling anchor -- quoting stress at the local battle-range
edge, faster resolution clock.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100          # 100 x 3s rows = 300s windows (rolling range, spread z)
EPS = 1e-12


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(roll_range_pos - 0.5) x z(quoted_spread_ticks, 300s)."""
    mid = pl.col("mid_px")
    rmax = mid.rolling_max(window_size=W, min_samples=W)
    rmin = mid.rolling_min(window_size=W, min_samples=W)
    rng = rmax - rmin
    pos = (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - rmin) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    cpos = pos - 0.5
    sp_z = _z(pl.col("quoted_spread_ticks"), W)
    return part.select((cpos * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="rollrpos_x_spread_z",
    mechanism=(
        "Local-anchor variant of the admitted champion interaction. "
        "range_pos_x_spread_z passed four horizons (900s IC +0.27): "
        "unusually wide quoting at the DAY envelope flags makers "
        "treating the boundary as toxic, and the boundary resolves in "
        "the direction it presses. The SAME logic at the ROLLING 300s "
        "range edge runs on a faster clock: local boundaries recycle "
        "many times per day, and stressed quoting at a freshly contested "
        "local edge marks imminent resolution decisions by scalpers "
        "rather than position traders. Centered rolling position x "
        "spread z carries POSITIVE IC at 30-300s. Falsifiable: if "
        "local-edge stress instead precedes REJECTION (negative IC), "
        "the toxicity reading does not transfer from day boundaries to "
        "local ones. Dedup: the rolling anchor drops stale extremes and "
        "decorrelates from the cumulative day position parent, and the "
        "faster target horizons separate it from the parent's 300-900s "
        "passes -- a change of question (which boundary, which clock), "
        "not a window tweak."
    ),
    info_set="mid_px, quoted_spread_ticks",
    inspiration=(
        "iter-003 R3-D family brief direction 6 (spread_z x range_pos "
        "exists -- vary the question); round-2 admitted "
        "range_pos_x_spread_z x mid_roll_range_pos_300s."
    ),
    compute=compute,
)
