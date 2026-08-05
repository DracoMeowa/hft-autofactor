"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

netvol_accum_300s: trailing 300s mean of SIGNED RAW aggressive volume
(buy_vol_60s - sell_vol_60s) -- the magnitude-weighted cousin of
ratio-based flow accumulators.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing accumulation window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 300s mean of (buy_vol_60s - sell_vol_60s); warm-up null."""
    x = pl.col("buy_vol_60s") - pl.col("sell_vol_60s")
    return part.select(
        x.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="netvol_accum_300s",
    mechanism=(
        "Every ratio-based flow accumulator (trade_imbalance, hence the "
        "registered ti_accum_300s) normalizes signed volume by total "
        "volume and throws away the MAGNITUDE dimension: a 60/40 buy/sell "
        "split scores the same whether it trades 1e3 or 1e6 units. Market "
        "impact does not work that way -- the square-root impact law "
        "(Bouchaud et al. 2004; Almgren-Chriss) ties price displacement to "
        "the absolute signed volume worked, so imbalance concentrated in "
        "high-participation phases (opening/closing auctions, information "
        "episodes) moves price far more than the same ratio on a thin "
        "tape. The trailing-300s mean of raw signed aggressive volume "
        "(buy_vol_60s - sell_vol_60s, batch-2 columns) restores exactly "
        "that magnitude weighting: large positive values mean substantial "
        "one-sided volume was actually transacted recently -> continued "
        "drift in that direction at 300-900s while the meta-order "
        "completes. Note the brief's literal '(buy-sell)/(buy+sell)' "
        "ratio is BY ENGINE DEFINITION identical to trade_imbalance_60s "
        "and its accumulator would clone ti_accum_300s (rho ~1); the raw "
        "signed volume is the dedup-safe realization of the same idea."
    ),
    info_set="buy_vol_60s, sell_vol_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 2 (net buy-sell pressure "
        "accumulation), adjusted after reading the engine source "
        "(factors_tick.cpp: TradeImbalanceWindow and SideVol60s use the "
        "same TrdBSFlag attribution, so the ratio equals "
        "trade_imbalance_60s). Magnitude-weighted flow per the impact "
        "power law."
    ),
    compute=compute,
)
