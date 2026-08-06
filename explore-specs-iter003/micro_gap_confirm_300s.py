"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

micro_gap_confirm_300s: interaction -- regime-adjusted microprice pressure x
regime-adjusted aggressor gap. Book pressure and last-trade aggressor
pointing the same way = confirmed short-horizon direction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment (588000)
W = 100       # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(microprice_dev, 300s) x z(gap_ticks, 300s); warm-up null."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    micro_z = _z(pl.col("microprice_dev"), W)
    gap_z = _z(gap_ticks, W)
    return part.select((micro_z * gap_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="micro_gap_confirm_300s",
    mechanism=(
        "Queue-pressure confirmation by the tape: microprice_dev is the "
        "PASSIVE read of direction (quantity-weighted touch position -- "
        "which queue is heavier), the aggressor gap is the ACTIVE read "
        "(where the last trade actually crossed). When the touch pressure "
        "and the last aggressor point the SAME way and both are unusually "
        "elevated vs their own trailing-300s regimes, the passive queue "
        "and the active flow are aligned -- a two-channel directional "
        "commitment that continues at 15-60s. When they conflict (the "
        "print crossed AGAINST the heavier queue), the trade fought the "
        "book's imbalance: either it exhausts against the queue or it was "
        "a stale/liquidity-taking print, and the gap tends to snap back "
        "toward the queue side. The product of the two z-scores is "
        "positive in the aligned cell and negative in the conflict cell; "
        "the two legs measure different physical objects (queue stock vs "
        "trade event), so the confirmation is a genuine cross-channel "
        "statement, not a reskin of either leg."
    ),
    info_set="microprice_dev, last_px, mid_px",
    inspiration=(
        "iter-003 R4-D brief direction (d) cross-column quote-shape "
        "structure; Stoikov (2018) microprice as next-move predictor "
        "combined with Lee & Ready (1991) aggressor classification; "
        "interaction form avoids the dead bare-microprice clones."
    ),
    compute=compute,
)
