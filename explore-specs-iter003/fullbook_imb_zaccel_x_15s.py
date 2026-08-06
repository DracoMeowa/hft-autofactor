"""Explore-lane prototype spec (iter-003 R6A, family R6A).

fullbook_imb_zaccel_x_15s: z-level crossed with acceleration DIRECTION on
the full-book imbalance. z_300(full-book imbalance) * sign(d2z). The broad-
book regime level gated by whether its own 15s acceleration confirms
(building) or opposes (fading) the level.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback


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


def compute(part: pl.DataFrame) -> pl.Series:
    """z * sign(d2z) where d2z = 15s acceleration of the full-book z.

    Warm-up rows null (z warm-up propagates through two shifts; sign of
    null acceleration is null, making the product null).
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((z * d2z.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zaccel_x_15s",
    mechanism=(
        "Broad-book imbalance level gated by acceleration direction: "
        "z_300(full-book imbalance) * sign(d2z). The full-book imbalance "
        "carries passive institutional tilt including hidden depth beyond "
        "level 5; its z-level alone is slow regime state. The acceleration "
        "DIRECTION (binary sign of d2z, the curvature of the z-trajectory) "
        "isolates inflection TIMING: when the bullish broad tilt is "
        "positive and acceleration is also positive, deep passive "
        "positioning is still INTENSIFYING -> mid drifts up at 15-60s; "
        "when the level is positive but acceleration turned negative, the "
        "broad regime peaked and the overextended tilt reverts. The "
        "binary sign discards magnitude, isolating whether the broad-book "
        "rate of change is still curving in the regime direction -- a "
        "second-order build-vs-exhaust question that velocity (1st "
        "derivative) cannot answer: a regime can have high positive "
        "velocity but zero acceleration (steady drift, no new commitment) "
        "and scores 0 here."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6A family brief: z-crossed-with-acceleration-sign "
        "construction on the full-book imbalance base; the acceleration "
        "direction gate is a different economic question than the product "
        "form on the same base."
    ),
    compute=compute,
)
