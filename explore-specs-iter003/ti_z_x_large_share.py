"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ti_z_x_large_share: signed aggressive flow weighted by the share of volume
arriving in large prints -- flow that comes in big tickets is informed flow.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s trailing z-score window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(trade_imbalance_60s, 180s) x large_trade_share_60s level weight."""
    ti_z = _z(pl.col("trade_imbalance_60s"), W)
    share = pl.col("large_trade_share_60s")
    return part.select((ti_z * share).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_z_x_large_share",
    mechanism=(
        "Informed-flow conditioning: the same net aggressive imbalance "
        "carries different information depending on the size distribution "
        "that produced it. Net buying imbalance assembled from a few large "
        "prints flags institutional/informed execution (information events "
        "do not get split to dust), while the same imbalance from many "
        "small retail prints is noise. Multiplying the 180s z-scored signed "
        "imbalance by the concurrent large-trade volume share up-weights "
        "the informed variant, which should continue at 15-60s horizons. "
        "This is the signed/conditioned repair of the direction-free "
        "large-trade level that died in iter-002."
    ),
    info_set="trade_imbalance_60s, large_trade_share_60s (wishlist)",
    inspiration=(
        "iter-002 lesson: large_trade_share_level IC ~= 0 direction-free, "
        "while the flow_divergence_300s champion shows SIGNED conditioned "
        "flow works short-horizon; Zhang & Shen (2018) trade-size "
        "decomposition of price impact; Bouchaud et al. (2009) anomalous "
        "impact of large trades."
    ),
    compute=compute,
)
