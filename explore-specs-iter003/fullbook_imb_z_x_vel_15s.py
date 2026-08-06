"""Explore-lane prototype spec (iter-003 R5, family R5-B).

fullbook_imb_z_x_vel_15s: NEW construction -- level-velocity SIGN-ALIGNMENT
product on the full-book imbalance. z(fullbook_imb) * dz. Positive when
level and velocity agree (broad-book momentum confirmation), negative when
they oppose (reversal). Tests sign-alignment on the patient-positioning base.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback


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
    """z * dz on the full-book imbalance regime.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)
    return part.select((z * dz).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_z_x_vel_15s",
    mechanism=(
        "Broad-book level-velocity sign-alignment: z_300(fullbook_imb) * "
        "dz. Positive when the full-book imbalance level and its 15s "
        "velocity share sign -- the whole visible book is still building "
        "in one direction (broad bid dominance increasing, or ask "
        "dominance deepening) -- multi-level patient positioning that "
        "continues at 15-60s. Negative when level and velocity oppose -- "
        "the broad-book tilt peaked and reverts. The full-book base "
        "(batch-2 total_*_vol) captures deeper institutional limit "
        "interest invisible to top-5 depth, so sign-alignment here tests "
        "whether SUSTAINED broad positioning (not just touch pressure) "
        "confirms momentum. Distinct from library fullbook_imb_z_cross_vel_15s "
        "(round-4 sign-flip event) and fullbook_imb_zvel_extreme_15s "
        "(implied product dz*|z|): the sign source includes the LEVEL, so "
        "bearish broad-book deepening scores positive (continuation), not "
        "negative as in velocity-only products."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R5-B family brief: level-velocity sign-alignment "
        "construction on the broad-book base; tests whether full-book "
        "momentum confirmation (both signs aligned) carries signal."
    ),
    compute=compute,
)
