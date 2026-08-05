"""Explore-lane prototype spec (iter-003, etf-regime lens).

arrival_accel_x_ti: delta-of-z(n_trades_60s, 100 rows) x sign(trade
imbalance).  ARRIVAL ACCELERATION with direction -- the dead iter-002
trade_arrival_burst measured the LEVEL of burstiness; this measures its
30s CHANGE (episode onset vs ongoing episode) and signs it by aggressor flow.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 300s (100 x 3s rows) z window for the arrival intensity
Z_WINDOW = 100
#: 10 x 3s rows = 30s acceleration horizon
ACCEL_ROWS = 10


def compute(part: pl.DataFrame) -> pl.Series:
    """30s change of the arrival z, signed by flow; warm-up rows null."""
    n = pl.col("n_trades_60s")
    mean = n.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = n.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    z = (n - mean) / std
    z = pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(z)
    accel = z.diff(ACCEL_ROWS)
    ti = pl.col("trade_imbalance_60s")
    val = accel * ti.sign()
    return part.select(
        pl.when(accel.is_not_null() & ti.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="arrival_accel_x_ti",
    mechanism=(
        "Arrival acceleration marks episode ONSET: trade arrivals cluster "
        "(Hawkes self-excitation -- arrivals breed arrivals), so the "
        "informative moment of an information episode is when the arrival "
        "intensity is RISING relative to its recent norm, not when it is "
        "already elevated. The iter-002 level burst (trade_arrival_burst: "
        "z of 300s arrival vs 1800s baseline) died because a high level "
        "confounds fresh onsets with episodes already in progress (whose "
        "information is already in the price). The 30s delta of the "
        "arrival z isolates onsets; multiplying by the aggressor-flow sign "
        "converts direction-free intensity change into a signed signal: "
        "accelerating activity WITH the aggressive-flow direction is an "
        "unfolding informed episode -> continuation at 30-300s; "
        "deceleration (negative acceleration) flags exhaustion and "
        "expected fade of the just-revealed direction."
    ),
    info_set="n_trades_60s, trade_imbalance_60s",
    inspiration=(
        "iter-002 archive: trade_arrival_burst (arrival LEVEL burst) IC ~ "
        "0; iter-003 etf-regime brief: arrival ACCELERATION with "
        "direction. Hawkes (1971) self-excitation; Engle & Russell (1998) "
        "ACD; n_trades_60s materialized 2026-08-05 (wishlist)."
    ),
    compute=compute,
)
