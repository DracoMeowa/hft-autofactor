"""Explore-lane prototype spec (iter-003 R5, family R5-C).

fullbook_imb_zcross_x_wdi_zvel: broad-book crossing events WEIGHTED by
the depth-5 extreme velocity intensity. The fullbook imbalance crossing
velocity (event-sparse) multiplied by the wdi extreme velocity (continuous),
testing whether full-book regime changes backed by top-5 momentum survive
at longer horizons.
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
    """fullbook crossing velocity * wdi extreme velocity; warm-up null."""
    zf = _z(_fullbook_imb(), W)
    zf_lag = zf.shift(LAG)
    flip = (
        pl.when(zf.is_null() | zf_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((zf.sign() != zf_lag.sign()) & (zf != 0) & (zf_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    fb_cross_vel = flip * (zf - zf_lag)

    zw = _z(pl.col("wdi"), W)
    dzw = zw - zw.shift(LAG)
    wdi_zvel = dzw * zw.abs()

    return part.select((fb_cross_vel * wdi_zvel).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zcross_x_wdi_zvel",
    mechanism=(
        "Depth-confirmed broad-book crossings: a fullbook imbalance "
        "crossing (the entire patient book relocating) is the rarest and "
        "costliest book event -- but the crossing velocity alone (equal "
        "magnitude for a slow drift-across and a decisive flip) does not "
        "distinguish a genuine regime change from a marginal zero-crossing. "
        "Hypothesis: weighting each broad-book crossing by the CONCURRENT "
        "5-level depth extreme velocity (dz_wdi * |z_wdi|) ranks crossings "
        "by how much top-5 depth momentum backs them. A crossing where the "
        "5-level depth is simultaneously moving at extreme speed in the "
        "same direction is a synchronized full-market repositioning -- "
        "patient broad inventory AND visible top-5 depth both relocating "
        "at once -- the highest-conviction event, whose direction survives "
        "at 300-900s. A crossing where the 5-level depth is quiescent is "
        "just the deep tail shifting with no visible urgency, likely noise. "
        "The product is nonzero only at broad-book crossings but VARIES "
        "across them by depth backing, so it re-ranks crossing events -- "
        "not a reskin of either parent (the crossing base is event-sparse; "
        "the depth velocity is continuous; the product is neither alone)."
    ),
    info_set="total_bid_vol, total_ask_vol, wdi",
    inspiration=(
        "iter-003 R5-C family brief direction 5 (cross of TWO z-vel "
        "bases): broad-book crossing × depth-5 extreme velocity. Tests "
        "multi-scale synchronization between the full book and the "
        "visible top-5 depth."
    ),
    compute=compute,
)
