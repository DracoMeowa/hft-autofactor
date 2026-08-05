"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

large_net_accum_300s: trailing 300s mean of large_trade_net_share_60s --
the sustained net direction of the largest ~10% of trades (institutional
footprint regime).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing accumulation window


def compute(part: pl.DataFrame) -> pl.Series:
    """Trailing 300s mean of large_trade_net_share_60s; warm-up null."""
    x = pl.col("large_trade_net_share_60s")
    return part.select(
        x.rolling_mean(window_size=W, min_samples=W).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="large_net_accum_300s",
    mechanism=(
        "The largest ~10% of trades are disproportionately institutional "
        "/ informed (Barclay-Warner 1995: big trades carry information), "
        "and institutional execution is SCHEDULED: algorithms slice "
        "parent orders over many minutes, so once the big-print flow turns "
        "net one-sided it tends to STAY net one-sided for the life of the "
        "parent order. Averaging the signed net share over 300s measures "
        "whether such a footprint regime is in progress: sustained "
        "positive values = institutional net buying still being worked -> "
        "continuation at 300-900s horizons matched to execution "
        "timescales. Distinct from the dead large_share_mom_300s "
        "(IS-dead in round 1): that was the 300s DELTA of the UNSIGNED "
        "participation share (a regime-transition guess on a level); this "
        "is the ACCUMULATION of the SIGNED net share -- direction "
        "information the unsigned share never had."
    ),
    info_set="large_trade_net_share_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 2 (300s accumulation of "
        "large_trade_net_share_60s). Trade-size information content "
        "(Barclay & Warner 1995); meta-order persistence (Bouchaud et al. "
        "2004). Signed-net column is batch-2; avoids the dead unsigned-"
        "share level/momentum forms."
    ),
    compute=compute,
)
