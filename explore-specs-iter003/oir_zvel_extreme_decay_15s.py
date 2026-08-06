"""Explore-lane prototype spec (iter-003 R6C, family R6C).

oir_zvel_extreme_decay_15s: RECENCY-DECAY extremeness weighting on the
z-velocity product applied to the touch imbalance. The 15s z-velocity of
z_300(oir) weighted by a linearly-decayed |z| over a trailing 20-row (60s)
window: recent extreme touch regimes dominate the weight, so velocity is
amplified when the best-quote imbalance has been SUSTAINED-recently
extreme.
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
    sustained-recent extreme touch-imbalance regime amplifies the
    velocity more than a single-row flash spike.
    """
    z = _z(pl.col("oir"), W)
    dz = z - z.shift(LAG)
    decayed_absz = _wma(z.abs(), DECAY)
    return part.select((dz * decayed_absz).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zvel_extreme_decay_15s",
    mechanism=(
        "Recency-decayed touch-imbalance velocity: the 15s z-velocity of "
        "z_300(oir) weighted by a RECENTLY-DECAYED extremeness -- a 20-row "
        "(60s) linearly-weighted moving average of |z| (current row weight "
        "20, oldest weight 1). The touch imbalance (oir, best-quote "
        "quantity ratio) is the most flicker-prone book signal -- it "
        "changes on every single order placement or cancellation at the "
        "best price, so instantaneous |z| may be dominated by transient "
        "quote flicker rather than committed positioning. The decayed "
        "|z| filters that flicker: a touch regime must hold its extreme "
        "for multiple snapshots to accumulate decayed weight. The "
        "hypothesis: touch-imbalance velocity is more predictive when "
        "amplified by a COMMITTED-recent extreme (a bid- or ask-heavy "
        "touch that has persisted for 30-60s, indicating genuine "
        "positioning rather than a momentary quote refresh) than by the "
        "instantaneous |z|. Distinct from the threshold-gate R6C variants "
        "(hard binary gate): this is a SOFT recency weighting -- no row "
        "is zeroed, but stale extremes are down-weighted relative to "
        "recent ones."
    ),
    info_set="oir",
    inspiration=(
        "iter-003 R6C family brief: recency-decay variant of the proven "
        "z-velocity-extremity product on the touch imbalance; the "
        "flicker-prone nature of oir makes recency filtering "
        "economically motivated."
    ),
    compute=compute,
)
