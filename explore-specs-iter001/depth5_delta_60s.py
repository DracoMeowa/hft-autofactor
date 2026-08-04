"""Explore-lane prototype spec (iter-001, flow-queue lens).

depth5_delta_60s: 60s change (delta) of 5-level book imbalance.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

D = 20  # 20 x 3s rows = 60s delta


def compute(part: pl.DataFrame) -> pl.Series:
    b = pl.col("depth_bid5").cast(pl.Float64)
    a = pl.col("depth_ask5").cast(pl.Float64)
    tot = b + a
    imb = (
        pl.when(tot > 0.0)
        .then((b - a) / tot)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select(imb.diff(D).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="depth5_delta_60s",
    mechanism=(
        "Depth delta: the trailing-60s CHANGE in 5-level depth imbalance "
        "(bid-ask)/(bid+ask), not its level. A book whose bid-side depth "
        "stock is actively growing relative to the ask side is experiencing "
        "net limit-order inflow on the bid - latent demand building up "
        "before it shows up in trades. The level of imbalance (wdi) is "
        "crowded and slow; the delta captures the flow INTO the deeper "
        "book and should decorrelate from the level by construction "
        "(differencing removes the persistent state), targeting the "
        "digest's explicitly unexplored 'depth delta' dimension."
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "Digest iter-000 opportunity map: 'Depth-side unexplored: "
        "resiliency, depth delta, queue position, large-order depth share'; "
        "cross-family mutation away from the depth mega-family has higher "
        "marginal value (rho 0.69 within family); Cao-Hansch-Wang (2009) "
        "information content of the book's depth dynamics."
    ),
    compute=compute,
)
