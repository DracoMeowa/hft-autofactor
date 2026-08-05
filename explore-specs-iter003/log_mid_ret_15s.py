"""Explore-lane prototype spec (iter-003, price-vol family).

log_mid_ret_15s: fastest signed mid-price momentum (15s = 5 x 3s rows).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 5 rows x 3s = 15s trailing window
K = 5


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing-15s log-mid return; warm-up rows null (diff semantics)."""
    return part.select(pl.col("mid_px").log().diff(K).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="log_mid_ret_15s",
    mechanism=(
        "Fastest-window signed mid momentum: the built-in log_mid_ret_60s "
        "measures drift over a full minute, but on liquid ETFs the "
        "information that moves the mid is often absorbed within a few "
        "snapshots. A 15s (5-row) trailing return isolates the most recent "
        "directional impulse before it is averaged away, testing whether "
        "the fastest moves continue (momentum from aggressive information "
        "flow) or snap back (bid-ask bounce / inventory mean-reversion). "
        "Distinct from the 60s window by construction: shorter windows "
        "weight the latest tick more heavily and decorrelate as the "
        "horizon mismatch grows. Targets the 15s/30s horizons where "
        "fast-moving state is what carries."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 price-vol family brief: 'momentum exists in the built-in "
        "log_mid_ret_60s family -- vary windows'. Short-horizon return "
        "autocorrelation (Lo & MacKinlay 1990); signed/fast companion of the "
        "engine's log_mid_ret_60s and of the unsigned rv_60s."
    ),
    compute=compute,
)
