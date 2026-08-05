"""Explore-lane prototype spec (iter-003, price-vol family).

log_mid_ret_30s: fast signed mid-price momentum (30s = 10 x 3s rows).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: 10 rows x 3s = 30s trailing window
K = 10


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing-30s log-mid return; warm-up rows null (diff semantics)."""
    return part.select(pl.col("mid_px").log().diff(K).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="log_mid_ret_30s",
    mechanism=(
        "Fast signed mid momentum at the 30s scale: sits between the "
        "15s impulse and the built-in 60s drift, a window long enough to "
        "average out single-snapshot bid-ask bounce yet short enough to "
        "still reflect a single directional episode. A persistent 30s move "
        "flags informed order flow still being worked (continuation), while "
        "a 30s move that has already exhausted itself flags mean reversion. "
        "Window variation across 15/30/60/120s lets the screen discover "
        "which impulse horizon the ETF's microstructure actually carries."
    ),
    info_set="mid_px",
    inspiration=(
        "iter-003 price-vol family brief: 'momentum exists in the built-in "
        "log_mid_ret_60s family -- vary windows'. Signed companion of the "
        "engine's rv/return library; Lo & MacKinlay (1990) short-horizon "
        "autocorrelation."
    ),
    compute=compute,
)
