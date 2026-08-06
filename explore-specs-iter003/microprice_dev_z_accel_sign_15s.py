"""Explore-lane prototype spec (iter-003 R5, family R5-B).

microprice_dev_z_accel_sign_15s: NEW construction variant of the
z-vs-velocity template on microprice_dev -- LEVEL signed by ACCELERATION
direction. z * sign(d2z). Tests whether the microprice-deviation regime's
acceleration direction (not velocity direction) gates the level's
predictive content. When acceleration confirms the level (both up, or
both down), the queue-pressure edge is still building -> continuation;
when acceleration opposes the level, the edge peaked -> reversion.
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


def compute(part: pl.DataFrame) -> pl.Series:
    """z * sign(d2z) where d2z = 15s acceleration of z.

    Warm-up rows null (z warm-up propagates through two shifts; sign of
    null acceleration is null, making the product null).
    """
    z = _z(pl.col("microprice_dev"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((z * d2z.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="microprice_dev_z_accel_sign_15s",
    mechanism=(
        "Microprice-deviation level signed by acceleration direction: "
        "z_300(microprice_dev) * sign(d2z). microprice_dev (micro minus "
        "mid, in px) is the queue-weighted fair-value lead -- positive "
        "means the heavy side is the bid queue. Its z-level alone is dead "
        "(round-3 microprice_dev_z_300s rejected); the round-4 velocity "
        "product (dz * |z|) was admitted. This construction asks a "
        "different question: does the DIRECTION OF ACCELERATION (not "
        "velocity) gate the level? When the bullish deviation is positive "
        "and acceleration is also positive (d2z > 0), the queue pressure "
        "is still INTENSIFYING -> mid follows the deviation up. When the "
        "level is positive but acceleration turned negative, the pressure "
        "peaked and is fading -> mean reversion. The binary sign(d2z) "
        "isolates inflection TIMING that velocity alone misses: a regime "
        "can have high positive velocity but zero acceleration (steady "
        "build, no new information) -- scored 0 here."
    ),
    info_set="microprice_dev",
    inspiration=(
        "iter-003 R5-B family brief: z-signed-by-acceleration-direction "
        "construction; tests a NEW economic question (acceleration "
        "direction as level gate) on a base whose level-z and velocity-z "
        "are already characterized."
    ),
    compute=compute,
)
