"""Explore-lane prototype spec (iter-003 R5, family R5-A).

rv60_logzvel_div_60s: z-level vs instantaneous-velocity divergence on
log(rv_60s), SIGNED-DIFFERENCE form -- the slow vol-regime z minus its
own fast z-velocity, itself regime-normalized.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 20  # 20 x 3s rows = 60s velocity lookback


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
    """z(log(rv_60s), 300s) - z(dz, 300s) where dz = 60s z-velocity.

    Warm-up rows null: z warm-up propagates through dz into the
    velocity's own trailing z. Both terms regime-normalized.
    """
    z_e = _z(_log_rv(pl.col("rv_60s")), W)
    dz_e = z_e - z_e.shift(LAG)
    tmp = part.select(z_e.alias("_z"), dz_e.alias("_dz"))
    tmp = tmp.select(pl.col("_z"), _z(pl.col("_dz"), W).alias("_dzz"))
    return tmp.select((pl.col("_z") - pl.col("_dzz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="rv60_logzvel_div_60s",
    mechanism=(
        "Vol level-vs-velocity overextension: z_300(log rv_60s) minus "
        "the trailing-300s z of its own 60s z-velocity. When short-term "
        "vol reads extreme but its 60s velocity is already normalizing "
        "(large positive gap: level stretched, velocity fading), the vol "
        "spike is exhausting -- the information event that drove the "
        "spike has been absorbed, and price stabilizes or mean-reverts "
        "at 60-300s. When normalized velocity leads the level (large "
        "negative gap: level still moderate but velocity surging), the "
        "vol spike is still building and further directional drift "
        "follows. Both components are z-normalized against their own "
        "300s distributions, making this a relative-stretch measure -- "
        "it asks whether the vol regime is building or exhausting "
        "relative to its own dynamics. DEDUP: library rv_z_300s is the "
        "pure LEVEL z (no velocity component); library signed_rv_60s is "
        "a from-scratch signed rv (different input). The 60s velocity "
        "timescale captures the vol-of-vol dynamics that the 15s "
        "variant would miss. The log transform prevents tail spikes from "
        "dominating the z, unlike the library rv_z_300s which uses raw "
        "rv_300s."
    ),
    info_set="rv_60s",
    inspiration=(
        "iter-003 R5-A family brief: signed-divergence form of the "
        "ofi_z_cross_vel_15s z-vs-velocity template on log(rv_60s) with "
        "a 60s velocity; the div construction isolates vol-regime "
        "build/exhaust tension."
    ),
    compute=compute,
)
