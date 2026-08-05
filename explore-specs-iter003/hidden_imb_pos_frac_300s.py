"""Explore-lane prototype spec (iter-003 R3-B, deep-book divergence lens).

hidden_imb_pos_frac_300s: trailing-300s fraction of snapshots where the
hidden-layer imbalance is POSITIVE -- persistence/duration of the hidden
bid-skew regime, an occupancy statistic orthogonal to the magnitude z.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window


def _hidden_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    hb = pl.when(tb > db).then(tb - db).otherwise(pl.lit(0.0))
    ha = pl.when(ta > da).then(ta - da).otherwise(pl.lit(0.0))
    den = hb + ha
    # Zero hidden depth on both sides -> neutral 0.0 (no hidden skew). Kept
    # a real value (not null) so the trailing occupancy window is complete.
    return pl.when(den > 0.0).then((hb - ha) / den).otherwise(pl.lit(0.0))


def compute(part: pl.DataFrame) -> pl.Series:
    """Fraction of last-300s rows with hidden-layer imbalance > 0."""
    himb = _hidden_imb()
    pos = (
        pl.when(himb.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(himb > 0.0)
        .then(1.0)
        .otherwise(0.0)
    )
    frac = pos.rolling_mean(window_size=W, min_samples=W)
    return part.select(frac.alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="hidden_imb_pos_frac_300s",
    mechanism=(
        "Hidden-regime duration: the share of the trailing 300s during which "
        "the hidden layer (depth beyond the top-5) is net BID-skewed. "
        "hidden_imb_z_300s measures how strongly hidden depth leans one way "
        "right now; this occupancy measures how CONSISTENTLY it has leaned "
        "that way. A persistently hidden-bid-skewed queue is durable patient "
        "accumulation -- demand that has stayed parked below the touch across "
        "many snapshots, not a transient flicker -- and such entrenched "
        "positioning decays slowly, conditioning the next minutes toward "
        "upward drift / a firm floor at 300-900s. Duration and intensity are "
        "different economic questions: a weak-but-constant hidden skew and a "
        "strong-but-oscillating one can share a z-extreme yet imply very "
        "different commitment. Occupancy is near-orthogonal to the z-score "
        "and to every momentum factor, in the slow regime class that paid "
        "off at long horizons."
    ),
    info_set="total_bid_vol, total_ask_vol, depth_bid5, depth_ask5 (batch-2)",
    inspiration=(
        "iter-003 R3-B brief direction 5 (hidden-depth imbalance, persistence "
        "facet); occupancy-vs-magnitude decomposition used successfully for "
        "the touch-vs-queue divergence; slow-regime horizons per eval-v2."
    ),
    compute=compute,
)
