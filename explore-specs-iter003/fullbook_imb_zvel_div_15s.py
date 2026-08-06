"""Explore-lane prototype spec (iter-003 R4, family R4-C).

fullbook_imb_zvel_div_15s: z-level vs instantaneous-velocity divergence on
the full-book imbalance, SIGNED-DIFFERENCE form -- the slow broad-book
regime z minus its own fast z-velocity, itself regime-normalized.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(fbi, 300s) - z(dz, 300s) where dz = 15s z-velocity.

    Warm-up rows null: z warm-up propagates through dz into the
    velocity's own trailing z.
    """
    z_e = _z(_fullbook_imb(), W)
    dz_e = z_e - z_e.shift(LAG)
    tmp = part.select(z_e.alias("_z"), dz_e.alias("_dz"))
    tmp = tmp.select(pl.col("_z"), _z(pl.col("_dz"), W).alias("_dzz"))
    return tmp.select((pl.col("_z") - pl.col("_dzz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zvel_div_15s",
    mechanism=(
        "Broad-book overextension vs its own fast edge: z_300(full-book "
        "imbalance) minus the trailing-300s z of its own 15s z-velocity. "
        "Deep-book tilt builds slowly (creation/redemption flows, "
        "patient institutional queue placement), so when the slow z reads "
        "extreme but the fast edge is already moving the other way, the "
        "patient tilt is being withdrawn or overrun RIGHT NOW and price "
        "drifts against the level's direction at 15-60s; when normalized "
        "velocity leads level, the broad regime is still building and "
        "continues. DEDUP: differs from library fullbook_imb_z_300s (pure "
        "level z -- regime state only) and from library "
        "fullbook_imb_mom_60s (raw unnormalized 60s delta): here the "
        "fast edge is regime-normalized and subtracted from the level, a "
        "relative-stretch question neither parent asks."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R4-C family brief: signed-divergence form of the "
        "admitted ofi_z_cross_vel_15s z-vs-velocity template applied to "
        "the full-book imbalance; slow-build nature of deep-book tilt "
        "makes level-vs-velocity tension a natural overextension probe."
    ),
    compute=compute,
)
