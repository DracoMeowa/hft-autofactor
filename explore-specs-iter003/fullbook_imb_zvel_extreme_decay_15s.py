"""Explore-lane prototype spec (iter-003 R6C, family R6C).

fullbook_imb_zvel_extreme_decay_15s: RECENCY-DECAY extremeness weighting on
the z-velocity product applied to the FULL-BOOK imbalance. The 15s
z-velocity of z_300(full-book imbalance) weighted by a linearly-decayed |z|
over a trailing 20-row (60s) window. The broad book builds slowly
(institutional), so a decayed extremeness measure matches its timescale
better than the instantaneous |z|.
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


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (sum); null when denominator is 0."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


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

    The 20-row linear decay kernel matches the slow build-time of broad-
    book institutional positioning: sustained-recent hidden-depth
    extremes amplify the velocity more than transient spikes.
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)
    decayed_absz = _wma(z.abs(), DECAY)
    return part.select((dz * decayed_absz).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zvel_extreme_decay_15s",
    mechanism=(
        "Recency-decayed broad-book velocity: the 15s z-velocity of "
        "z_300(full-book imbalance) weighted by a RECENTLY-DECAYED "
        "extremeness -- a 20-row (60s) linearly-weighted moving average "
        "of |z| (current row weight 20, oldest weight 1). The full-book "
        "imbalance (including depth beyond level 5) builds SLOWLY: "
        "institutional creation/redemption flows and patient passive "
        "placement accumulate over tens of seconds, so the instantaneous "
        "|z| may lag the true positioning state. The decayed |z| matches "
        "this timescale -- a broad regime that has held an extreme for "
        "30-60s (sustained institutional tilt) accumulates full decayed "
        "weight, while a flash extreme (one snapshot then reversion, "
        "likely feed artifact or transient MM refresh) contributes little. "
        "The hypothesis: broad-book velocity is more predictive when "
        "amplified by a SUSTAINED-recent extreme than by the "
        "instantaneous |z|, because the slow-build nature of deep-book "
        "positioning means commitment is expressed through persistence, "
        "not spike. Distinct from the wdi/oir decay variants: the full "
        "book aggregates hidden depth whose dynamics (slow, institutional) "
        "differ structurally from the top-5 (fast, retail/MM-driven)."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6C family brief: recency-decay variant of the proven "
        "z-velocity-extremity product on the full-book imbalance; the "
        "slow institutional build-time of the broad book makes the decay "
        "kernel a natural timescale match."
    ),
    compute=compute,
)
