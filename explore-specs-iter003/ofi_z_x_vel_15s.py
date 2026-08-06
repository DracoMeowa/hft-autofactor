"""Explore-lane prototype spec (iter-003 R5, family R5-B).

ofi_z_x_vel_15s: NEW construction -- level-velocity SIGN-ALIGNMENT product.
z(ofi_60s) * dz, where dz is the 15s z-velocity. Positive when level and
velocity point the same way (momentum confirmation), negative when they
oppose (momentum reversal). Tests sign-alignment as the economic input,
unlike zvel_extreme (dz * |z|) which carries sign from velocity only.
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


def compute(part: pl.DataFrame) -> pl.Series:
    """z * dz where dz = 15s z-velocity.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(pl.col("ofi_60s"), W)
    dz = z - z.shift(LAG)
    return part.select((z * dz).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_z_x_vel_15s",
    mechanism=(
        "Level-velocity sign-alignment product: z_300(ofi_60s) * dz. "
        "Positive when the OFI regime level and its 15s velocity have the "
        "SAME sign -- bullish book flow still building (z > 0, dz > 0) or "
        "bearish book flow still deepening (z < 0, dz < 0) -- pure "
        "momentum confirmation whose direction continues at 15-60s. "
        "Negative when level and velocity oppose -- the regime peaked and "
        "is reverting. Economically distinct from library ofi_z_cross_vel_15s "
        "(round-4, z sign-flip EVENT scored by velocity magnitude) and "
        "ofi_zvel_extreme_15s (dz * |z|, sign from velocity only): here "
        "the sign comes from BOTH level and velocity, so a bearish regime "
        "deepening (z < 0, dz < 0) scores POSITIVE (continuation) rather "
        "than negative. This tests whether momentum CONFIRMATION "
        "(alignment of level direction and velocity direction) is the "
        "live signal, vs mean-reversion of extreme levels. Because "
        "sign(z*dz) = sign(z) * sign(dz) flips with z's sign while "
        "zvel_extreme's sign = sign(dz) alone, the two diverge on roughly "
        "half the rows (where z < 0)."
    ),
    info_set="ofi_60s",
    inspiration=(
        "iter-003 R5-B family brief: level-velocity sign-alignment "
        "construction; a genuinely different sign source (both level and "
        "velocity) vs the round-4 velocity-only products."
    ),
    compute=compute,
)
