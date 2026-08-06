"""Explore-lane prototype spec (iter-003 R4, family R4-C).

microprice_dev_zvel_extreme_15s: z-level vs instantaneous-velocity
divergence on microprice_dev, PRODUCT form -- the 15s z-velocity of the
micro-deviation regime weighted by the extremity |z| of the regime.
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


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 15s z-velocity of the micro-deviation regime.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(pl.col("microprice_dev"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="microprice_dev_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted velocity of the microprice-deviation regime: "
        "the 15s change rate of z_300(microprice_dev) weighted by |z|. "
        "microprice_dev (micro minus mid, bps) is the queue-weighted fair-"
        "value lead: positive deviation means the heavy side is the BID "
        "queue (upward pressure priced in before mid moves). When an "
        "already extreme deviation regime accelerates further, the "
        "queue-pressure edge is being pushed decisively and mid follows "
        "the deviation direction at 15-60s; the same velocity near a "
        "neutral regime is quote noise and scores ~0 through the |z| "
        "weight. This is a level-x-velocity interaction, direction "
        "carried by the velocity: not the dead microprice_dev_z_300s "
        "level-z (round-3 rejected stem, bare level) and not the dead "
        "microprice_dev_mom_60s (raw unnormalized momentum) -- the "
        "extremity weighting is a different economic question than "
        "either: how hard an ALREADY-STRETCHED micro-imbalance is moving."
    ),
    info_set="microprice_dev",
    inspiration=(
        "iter-003 R4-C family brief: product form of the admitted "
        "ofi_z_cross_vel_15s z-vs-velocity template applied to the "
        "microprice-deviation state column; both prior bare transforms "
        "of this base died, motivating the interaction form."
    ),
    compute=compute,
)
