"""Explore-lane prototype spec (iter-003 R5, family R5-A).

rv300_logzvel_div_15s: z-level vs instantaneous-velocity divergence on
log(rv_300s), SIGNED-DIFFERENCE form -- the slow vol-regime z minus its
own fast 15s z-velocity, itself regime-normalized.
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


def _log_rv(x: pl.Expr) -> pl.Expr:
    """Safe log of rv: clips to positive minimum to avoid -inf on zero."""
    return x.cast(pl.Float64).clip(1e-20, 1e10).log()


def compute(part: pl.DataFrame) -> pl.Series:
    """z(log(rv_300s), 300s) - z(dz, 300s) where dz = 15s z-velocity.

    Warm-up rows null: z warm-up propagates through dz into the
    velocity's own trailing z. Both terms regime-normalized.
    """
    z_e = _z(_log_rv(pl.col("rv_300s")), W)
    dz_e = z_e - z_e.shift(LAG)
    tmp = part.select(z_e.alias("_z"), dz_e.alias("_dz"))
    tmp = tmp.select(pl.col("_z"), _z(pl.col("_dz"), W).alias("_dzz"))
    return tmp.select((pl.col("_z") - pl.col("_dzz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="rv300_logzvel_div_15s",
    mechanism=(
        "Fast-edge disagreement with slow-vol level: z_300(log rv_300s) "
        "minus the trailing-300s z of its own 15s z-velocity. rv_300s is "
        "the 5-minute realized variance (slow, persistent), and its 15s "
        "z-velocity is the fastest edge of the vol regime. When the slow "
        "level reads extreme but the fast edge is already reversing "
        "(large positive gap: level stretched, velocity fading), the vol "
        "spike peaked within the 5-minute window and price stabilizes or "
        "mean-reverts at 15-60s; when normalized velocity leads level "
        "(large negative gap: level moderate but velocity surging), a "
        "new vol spike is igniting and directional drift follows. The "
        "15s velocity on a 300s base isolates the leading-edge tension "
        "that the 60s velocity smooths over -- the slow-fast gap is "
        "maximally informative when measured at the widest timescale "
        "separation. Both terms are z-normalized. DEDUP: library "
        "rv_z_300s is the pure LEVEL z of raw rv_300s (no velocity, no "
        "log); library rv_ratio_z_300s is the z of the rv_60s/rv_300s "
        "RATIO (a different construction). Here the log(rv_300s) z is "
        "tensioned against its OWN 15s velocity, a relative-stretch "
        "question neither parent asks."
    ),
    info_set="rv_300s",
    inspiration=(
        "iter-003 R5-A family brief: signed-divergence form of the "
        "z-vs-velocity template on log(rv_300s) with a 15s velocity; "
        "the wide slow-base/fast-edge separation maximizes the "
        "level-velocity tension signal."
    ),
    compute=compute,
)
