"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ti_sign_streak_30s: UNBROKEN aggression streak intensity -- when
trade_imbalance_15s has kept the same sign for >= 30s straight the factor
carries the live imbalance strength, otherwise 0.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

K = 10  # 10 consecutive same-sign transitions = streak of >= 33s


def compute(part: pl.DataFrame) -> pl.Series:
    """rolling_min(same-sign indicators, K) x trade_imbalance_15s.

    rolling_min over the K transition matches is 1.0 iff EVERY one of the
    last K transitions kept the sign (an unbroken streak); otherwise 0.
    Inside a live streak the value is the current imbalance itself (how
    hard the committed flow is pressing); outside streaks it is exactly
    0. Warm-up rows null, never zero-filled.
    """
    ti = pl.col("trade_imbalance_15s")
    sgn = ti.sign()
    sgn_lag = sgn.shift(1)
    same = (
        pl.when(sgn.is_null() | sgn_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((sgn == sgn_lag) & (sgn != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    streak = same.rolling_min(window_size=K, min_samples=K)
    return part.select((streak * ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_sign_streak_30s",
    mechanism=(
        "Unbroken aggression streaks mark a working meta-order: when the "
        "15s trade-imbalance sign survives 10 consecutive 3s transitions "
        "(>= 30s of one-sided marketable flow without a single sign "
        "reversal), the tape is most likely being worked by a single "
        "directional program slicing through time (Kyle 1985) rather "
        "than two-sided noise. Inside such a streak the CURRENT "
        "imbalance magnitude measures how hard the committed participant "
        "is pressing right now, so the factor carries sign x strength "
        "during live streaks and is exactly 0 the moment the streak "
        "breaks -- a working buy streak predicts 15-60s drift up, a "
        "sell streak drift down, and broken/choppy tape is silent. The "
        "sign-survival gate is the economic content: the same imbalance "
        "reading scores 0 after a single contrary snapshot because a "
        "broken streak no longer evidences a committed order. This "
        "structurally breaks rank correlation with panel "
        "trade_imbalance_60s (the zero mass plus 4x-faster resolution "
        "re-rank the rows), and it differs from the dead pos_frac "
        "fractions (share of positive rows over 300s): the streak "
        "requires UNBROKEN consecutive sign."
    ),
    info_set="trade_imbalance_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R3-C brief direction 1 (flow commitment as run "
        "statistics); Kyle (1985) meta-order slicing; the round-1 fast "
        "book-momentum cluster recast as a sign-survival gate on the "
        "new 15s trade-imbalance column."
    ),
    compute=compute,
)
