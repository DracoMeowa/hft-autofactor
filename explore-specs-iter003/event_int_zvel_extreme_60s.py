"""Explore-lane prototype spec (iter-003 R5, family R5-A).

event_int_zvel_extreme_60s: z-level vs instantaneous-velocity divergence
on book_event_intensity_60s, PRODUCT form -- the 60s z-velocity of the
information-arrival regime weighted by the extremity |z| of the regime.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 20  # 20 x 3s rows = 60s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 60s z-velocity of the event-intensity regime.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(pl.col("book_event_intensity_60s"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="event_int_zvel_extreme_60s",
    mechanism=(
        "Extremity-weighted information-arrival velocity: the 60s change "
        "rate of z_300(book_event_intensity_60s), weighted by |z|. Event "
        "intensity is the feed-events-per-second rate -- a direct "
        "readout of information arrival, decoupled from price. When the "
        "intensity regime is already extreme versus its 300s norm (high "
        "|z|: tape running unusually hot or cold) AND still accelerating "
        "over 60s, a self-exciting information cluster (Hawkes process) "
        "is intensifying: each event triggers more events, and the "
        "cluster has further to run. During intensifying hot-tape "
        "episodes, informed flow dominates and its directional imprint "
        "persists at 60-300s; during intensifying cold-tape episodes "
        "(extreme quiet getting quieter), liquidity providers withdraw "
        "further and spreads widen, creating mean-reversion "
        "opportunities. The 60s window captures the cluster-escalation "
        "timescale that the 15s variant would miss (self-excitation "
        "builds over tens of seconds). The extremity weight zeroes out "
        "routine tape-noise around the norm. DEDUP: round-2 lesson says "
        "event-intensity z LEVELS are dead (IS-dead or OOS collapse); "
        "library event_intens_z_300s is the pure LEVEL z (state only). "
        "Here the z-velocity weighted by level extremity is a "
        "fundamentally different object: it scores how hard an "
        "ALREADY-EXTREME tape regime is moving, not where the tape sits. "
        "The 60s velocity and the extremity weighting are both absent "
        "from the library factor."
    ),
    info_set="book_event_intensity_60s",
    inspiration=(
        "iter-003 R5-A family brief: extreme (product) form of the "
        "z-vs-velocity template on book_event_intensity_60s with a 60s "
        "velocity; round-4 showed the extreme construction was the "
        "goldmine; the 60s window matches Hawkes self-excitation "
        "buildup timescales."
    ),
    compute=compute,
)
