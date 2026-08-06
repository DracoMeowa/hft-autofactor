"""Explore-lane prototype spec (iter-003 R6C, family R6C).

wdi_zvel_extreme_decay_15s: RECENCY-DECAY extremeness weighting on the
z-velocity product. The 15s z-velocity of z_300(wdi) weighted not by the
INSTANTANEOUS |z| (as in the admitted wdi_zvel_extreme_15s) but by a
linearly-decayed |z| over a trailing 20-row (60s) window: recent extreme
regimes dominate the weight, so velocity is amplified when the depth
imbalance has been SUSTAINED-recently extreme, not just a single-row spike.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback
DECAY = 20  # 20-row (60s) linear decay kernel for |z|


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _wma(x: pl.Expr, d: int) -> pl.Expr:
    """Causal linearly-weighted moving average (recent-heavy) over d rows.

    weight[k] = (d - k) for shift k = 0..d-1 (current row gets weight d,
    oldest gets 1). Built arithmetically from shifts -- no ewm_mean
    (polars 1.43 ewm_mean NaN-poisons after leading nulls).
    """
    wt = d * (d + 1) // 2
    terms = [(d - k) * x.shift(k) for k in range(d)]
    num = terms[0]
    for t in terms[1:]:
        num = num + t
    return num / wt


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * decayed_|z| where decayed_|z| = wma(|z|, 20); warm-up null.

    The 20-row linear decay kernel emphasizes recent |z| values: a
    sustained-recent extreme depth-imbalance regime amplifies the
    velocity more than a single-row flash spike.
    """
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    decayed_absz = _wma(z.abs(), DECAY)
    return part.select((dz * decayed_absz).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_extreme_decay_15s",
    mechanism=(
        "Recency-decayed depth-imbalance velocity: the 15s z-velocity of "
        "z_300(wdi) weighted by a RECENTLY-DECAYED extremeness -- a 20-row "
        "(60s) linearly-weighted moving average of |z|, where the current "
        "row's |z| gets the highest weight (20) and the oldest gets weight "
        "1. The admitted wdi_zvel_extreme_15s uses the INSTANTANEOUS |z| "
        "as the amplifier (velocity weighted by how extreme the regime is "
        "RIGHT NOW); this variant asks whether a SUSTAINED-RECENT extreme "
        "is a better amplifier. A single-row flash to |z|=3 (one snapshot "
        "spike then reversion) contributes little to the decayed weight, "
        "while a regime that has held |z|>2 for 30-60s dominates it. The "
        "hypothesis: depth-imbalance velocity is more predictive when the "
        "regime has been COMMITTED-recently extreme (persistent crowded "
        "queue, genuine institutional positioning) rather than just "
        "spiked (which may be transient quote flicker). Distinct from "
        "wdi_zvel_2sig_extreme_15s (hard threshold gate): this is a SOFT "
        "recency weighting, not a binary gate -- every row contributes, "
        "weighted by how recently extreme the regime has been."
    ),
    info_set="wdi",
    inspiration=(
        "iter-003 R6C family brief: recency-decay variant of the proven "
        "z-velocity-extremity product; tests whether sustained-recent "
        "extremeness outperforms instantaneous extremeness as the "
        "velocity amplifier."
    ),
    compute=compute,
)
