"""Explore-lane prototype spec (iter-003 R5, family R5-A).

event_int_z_cross_vel_15s: z-level vs instantaneous-velocity divergence
on book_event_intensity_60s (feed events per second), CROSSING form --
the 300s z of event intensity crossed zero within the last 15s; value is
the z-velocity, only at crossings, else 0. Information-arrival regime
transition events.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the information-arrival regime transition.
    """
    z = _z(pl.col("book_event_intensity_60s"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((flip * (z - z_lag)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="event_int_z_cross_vel_15s",
    mechanism=(
        "Information-arrival regime transition events: the trailing-300s "
        "z of book_event_intensity_60s (feed events per second) crosses "
        "zero within 15s. Event intensity is a direct readout of the "
        "information-arrival rate, decoupled from price moves and the "
        "snapshot cadence. A zero-crossing of its 300s z means the tape "
        "has just transitioned from below-norm activity (quiet) to "
        "above-norm activity (heated) or vice versa -- the ONSET or "
        "OFFSET of an information cluster. Information arrives in self-"
        "exciting clusters (Hawkes process, Bacry et al. 2015): the "
        "transition from quiet to heated marks the onset of a cluster "
        "that has further to run, while the reverse marks exhaustion. "
        "The crossing VELOCITY scores how sharply the transition is "
        "occurring: a fast cross from quiet to heated marks a sudden "
        "information event (news, block execution), and during such "
        "episodes the tape is dominated by information-driven flow "
        "whose directional imprint keeps pushing price; a slow cross "
        "scores weak. Event-sparse (0 off crossings). DEDUP: round-2 "
        "lesson says event-intensity z LEVELS are dead (IS-dead or OOS "
        "collapse) -- but library event_intens_z_300s is the pure LEVEL "
        "z (state only, no velocity). Here only the regime-TRANSITION "
        "EVENT is scored, which is a different economic question: not "
        "'is the tape hot right now?' (level state, dead) but 'did the "
        "tape just shift from cold to hot?' (transition event, "
        "untested)."
    ),
    info_set="book_event_intensity_60s",
    inspiration=(
        "iter-003 R5-A family brief: apply the crossing template to "
        "book_event_intensity_60s; round-2 lesson that the level z is "
        "dead motivates the transition-event construction as a "
        "genuinely different economic question."
    ),
    compute=compute,
)
