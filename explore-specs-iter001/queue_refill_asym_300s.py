"""Explore-lane prototype spec (iter-001, flow-queue lens).

queue_refill_asym_300s: snapshot-delta resiliency (queue refill asymmetry).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100          # trailing 300s window of depletion events
L = 5            # refill measured 15s (5 rows) after the depletion step
MIN_EVENTS = 2   # need >=2 events per side to report an asymmetry


def compute(part: pl.DataFrame) -> pl.Series:
    bq = pl.col("bid1_qty").cast(pl.Float64)
    aq = pl.col("ask1_qty").cast(pl.Float64)

    # depletion steps: top-of-book quantity drops vs previous snapshot
    dep_b = bq.diff() < 0.0
    dep_a = aq.diff() < 0.0
    # refill observed L rows AFTER the event (past information at emit time)
    ref_b = bq.shift(-L) - bq
    ref_a = aq.shift(-L) - aq
    ev_b = dep_b & ref_b.is_not_null()
    ev_a = dep_a & ref_a.is_not_null()

    # aggregate events in [t-W-L, t-L]: rolling over trailing W rows, then
    # shift forward by L so an event's refill is fully realized before use
    num_b = (
        pl.when(ev_b).then(ref_b).otherwise(0.0)
        .rolling_sum(window_size=W, min_samples=W).shift(L)
    )
    den_b = ev_b.cast(pl.Float64).rolling_sum(window_size=W, min_samples=W).shift(L)
    num_a = (
        pl.when(ev_a).then(ref_a).otherwise(0.0)
        .rolling_sum(window_size=W, min_samples=W).shift(L)
    )
    den_a = ev_a.cast(pl.Float64).rolling_sum(window_size=W, min_samples=W).shift(L)
    scale = (bq + aq).rolling_mean(window_size=W, min_samples=W).shift(L)

    ok = (
        den_b.is_not_null() & den_a.is_not_null()
        & (den_b >= MIN_EVENTS) & (den_a >= MIN_EVENTS)
        & scale.is_not_null() & (scale > 0.0)
    )
    val = (num_b / den_b - num_a / den_a) / scale
    return part.select(
        pl.when(ok).then(val).otherwise(pl.lit(None, dtype=pl.Float64)).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="queue_refill_asym_300s",
    mechanism=(
        "Coarse resiliency / queue-refill asymmetry: after a top-of-book "
        "depth depletion step on one side, how much depth is back 15s later, "
        "averaged over depletion events in the trailing 300s, bid vs ask, "
        "normalized by average top-of-book size. A bid queue that refills "
        "faster than the ask queue means patient buy-side liquidity keeps "
        "re-arming: downward probes get absorbed and the next move is "
        "biased up (fast refill = temporary impact = mean reversion of the "
        "depleting move; slow/partial refill = informed consumption = "
        "continuation). This is a genuine DYNAMICS quantity - refill speed - "
        "orthogonal to depth-imbalance LEVELS (the oir/wdi family)."
    ),
    info_set="bid1_qty, ask1_qty",
    inspiration=(
        "Digest iter-000: 'arrival/cancel dynamics only testable via "
        "snapshot-delta reconstruction' on SSE and the opportunity map lists "
        "resiliency as unexplored depth-side; Obizhaeva & Wang (2013) "
        "resiliency after large trades; Foucault-Kadan-Kandel (2005) limit "
        "order book replenishment."
    ),
    compute=compute,
)
