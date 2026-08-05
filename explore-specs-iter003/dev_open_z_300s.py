"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

dev_open_z_300s: trailing-300s z-score of the open-deviation -- is the
current stretch from the open unusual vs TODAY'S recent stretch regime?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s regime window


def compute(part: pl.DataFrame) -> pl.Series:
    """z((mid-open)/open, 300s trailing); constant windows -> 0.0."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    m = dev.rolling_mean(window_size=W, min_samples=W)
    s = dev.rolling_std(window_size=W, min_samples=W)
    z = (dev - m) / s
    out = pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)
    return part.select(out.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="dev_open_z_300s",
    mechanism=(
        "Regime-relative overextension from the open. The LEVEL of the "
        "open-deviation (dev_from_open_bps, round-2 all-horizon winner) "
        "treats every bps of stretch equally, but 30bp from the open means "
        "something different after a morning that has already oscillated "
        "+/-40bp than after one pinned within +/-5bp: the z-score divides "
        "the current stretch by the day's own recent deviation volatility, "
        "measuring 'more stretched than this morning's regime justifies'. "
        "A high z flags an overextension freshly built relative to the "
        "local equilibrium -- inventory one-sidedness that market makers "
        "and open-referenced arbitrage (creation/redemption benchmarks) "
        "pull back from -- predicting reversion; a deeply negative z flags "
        "snap-back overshoots. The rolling normalization strips the slow "
        "trend component that dominates the level factor, so this asks a "
        "different question (regime surprise, not accumulated state) and "
        "should decorrelate from the library parent by construction."
    ),
    info_set="mid_px, open_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 1 (rolling z of the "
        "open-deviation) built on dev_from_open_bps (round-2, all 5 "
        "horizons, 900s OOS IC -0.4975); same regime-normalization recipe "
        "that turned raw ofi_15s into ofi_15s_z_120s (all 5 horizons)."
    ),
    compute=compute,
)
