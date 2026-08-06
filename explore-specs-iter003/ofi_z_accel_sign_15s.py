"""Explore-lane prototype spec (iter-003 R5, family R5-B).

ofi_z_accel_sign_15s: NEW construction variant of the z-vs-velocity
template on ofi_60s -- LEVEL signed by ACCELERATION direction.
z(ofi_60s) * sign(d2z). Tests whether order-flow-imbalance regime's
acceleration direction gates the level's predictive content.
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

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(pl.col("ofi_60s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((z * d2z.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_z_accel_sign_15s",
    mechanism=(
        "Order-flow-imbalance level signed by acceleration direction: "
        "z_300(ofi_60s) * sign(d2z). The OFI regime's z-level measures "
        "persistent one-sided book flow; its acceleration (d2z, the 2nd "
        "difference) measures whether the flow is intensifying or fading. "
        "When the level is extreme and acceleration confirms it (same "
        "sign), the book flow is still building -> continuation of the "
        "flow direction at 15-60s. When acceleration opposes the level, "
        "the flow peaked and the overextended level reverts. Economically "
        "distinct from library ofi_z_cross_vel_15s (round-4, z sign-flip "
        "EVENT scored by velocity) and ofi_sign_persist_60s (sign-run "
        "persistence): this is the point-in-time acceleration DIRECTION "
        "gating the level, a different second-order question. The binary "
        "sign discards magnitude entirely, isolating timing information."
    ),
    info_set="ofi_60s",
    inspiration=(
        "iter-003 R5-B family brief: z-signed-by-acceleration on the OFI "
        "base; the acceleration direction isolates inflection timing that "
        "the admitted velocity-product (ofi_z_cross_vel_15s) does not."
    ),
    compute=compute,
)
