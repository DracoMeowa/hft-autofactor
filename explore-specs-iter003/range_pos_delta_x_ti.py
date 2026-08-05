"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

range_pos_delta_x_ti: 300s range-position delta multiplied by current
aggressive-flow imbalance -- flow confirmation of runs through the
intraday range.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100        # 100 x 3s rows = 300s delta window
EPS = 1e-12    # below this range the position is undefined (flat day)


def _range_pos() -> pl.Expr:
    """Position of mid within the causal intraday high-low range, [0,1]."""
    mid = pl.col("mid_px")
    hi = mid.cum_max()
    lo = mid.cum_min()
    rng = hi - lo
    return (
        pl.when(rng.is_not_null() & (rng > EPS))
        .then((mid - lo) / rng)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """(range_pos(i) - range_pos(i-100)) x trade_imbalance_60s."""
    delta = _range_pos().diff(W)
    return part.select((delta * pl.col("trade_imbalance_60s")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="range_pos_delta_x_ti",
    mechanism=(
        "Runs through the intraday range are not all equal: a 300s rise "
        "toward the day high WITH aggressive buying still active (positive "
        "product) is a chase -- retail/inexperienced flow lifting into the "
        "stop cluster and resistance where informed counterparties and "
        "market makers absorb into strength (the round-1 champion showed "
        "rejection dominates on this instrument: negative IC of range-"
        "position level at every horizon). Flow-confirmed range runs are "
        "therefore predicted EXHAUSTION moves: high positive product -> "
        "down-drift (negative IC); high negative product (falling toward "
        "the day low under aggressive selling) -> up-drift as selling "
        "exhausts into support. Unconfirmed runs (TI ~ 0 while the range "
        "position drifts) score near zero by construction, separating "
        "flow-driven approaches from passive quote-walk. The interaction "
        "of the champion's delta with the strongest short-horizon "
        "information variable (aggressor flow) targets the 60-300s band."
    ),
    info_set="mid_px, trade_imbalance_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 5 (interaction variant "
        "of the champion's slow delta, correlation pulled apart from the "
        "level). Round-1 evidence: mid_day_range_pos negative IC all "
        "horizons (rejection regime); state-conditioned interactions "
        "survive where levels die."
    ),
    compute=compute,
)
