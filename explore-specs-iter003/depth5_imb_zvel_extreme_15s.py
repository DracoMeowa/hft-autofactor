"""Explore-lane prototype spec (iter-003 R4, family R4-C).

depth5_imb_zvel_extreme_15s: z-level vs instantaneous-velocity divergence
on the UNWEIGHTED top-5 depth imbalance, PRODUCT form -- the 15s z-velocity
of the regime weighted by the extremity |z| of the regime being moved.
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


def _depth5_imb() -> pl.Expr:
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    den = db + da
    return (
        pl.when(den > 0.0)
        .then((db - da) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 15s z-velocity of the top-5 imbalance regime.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(_depth5_imb(), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="depth5_imb_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted velocity of the UNWEIGHTED visible-depth "
        "regime: the 15s change rate of z_300((depth_bid5 - depth_ask5)/"
        "(sum)) weighted by |z|. The base deliberately differs from the "
        "engine's wdi, which applies exp(-k/2) level weights: the flat-"
        "weighted top-5 ratio is proportionally more sensitive to "
        "level 3-5 queue fills and pulls that wdi discounts, i.e. to the "
        "outer visible stack where larger resting orders sit. Fast motion "
        "of an extreme visible-depth state marks decisive rebuilding or "
        "abandonment of a crowded stack and continues at 15-60s in the "
        "motion's direction; motion around a neutral regime scores ~0. "
        "DEDUP: not library top5_book_div_z_300s (that is the top-5 "
        "CONCENTRATION ratio vs the full book -- a different ratio "
        "entirely), not library depth5_delta_60s (raw qty delta), not a "
        "wdi re-skin (weighting scheme is the economic input change)."
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 R4-C family brief: product form of the admitted "
        "ofi_z_cross_vel_15s z-vs-velocity template; the base choice "
        "exploits that panel depth_*5 totals allow a flat-weighted "
        "imbalance distinct from the exp-weighted engine wdi."
    ),
    compute=compute,
)
