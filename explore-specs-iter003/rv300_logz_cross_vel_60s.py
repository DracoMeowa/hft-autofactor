"""Explore-lane prototype spec (iter-003 R5, family R5-A).

rv300_logz_cross_vel_60s: z-level vs instantaneous-velocity divergence on
log(rv_300s), CROSSING form -- the 300s z of log(rv_300s) crossed zero
within the last 60s; value is the z-velocity, only at crossings, else 0.
Slow-vol regime reversal events; 60s velocity targets the 300-900s horizon.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 20  # 20 x 3s rows = 60s crossing lookback


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
    """(z_now - z_60s_ago) where sign(z) flipped over 60s, else 0.

    z is computed on log(rv_300s) to tame the right-skew of variance.
    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the slow-vol regime transition.
    """
    z = _z(_log_rv(pl.col("rv_300s")), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((flip * (z - z_lag)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="rv300_logz_cross_vel_60s",
    mechanism=(
        "Slow-vol regime reversal events: the trailing-300s z of "
        "log(rv_300s) crosses zero within 60s. rv_300s is the 5-minute "
        "realized variance -- a slow, persistent vol measure. A zero-"
        "crossing of its 300s z means the 5-minute vol has transitioned "
        "from below-norm to above-norm (or vice versa) over the last "
        "minute, marking a structural vol regime change, not transient "
        "noise. The 60s crossing velocity measures how decisively the "
        "5-minute vol regime is shifting. Slow-vol transitions carry "
        "directional information at 300-900s horizons: vol clustering "
        "(Engle 1982) means a regime shift persists, and the leverage "
        "effect (price drops induce vol spikes) links rising vol to "
        "recent negative returns. The SIGNED velocity distinguishes vol "
        "rising (often post-drop, predictive of further drift) from vol "
        "falling (stabilization). Event-sparse (0 off crossings). "
        "DEDUP: library rv_z_300s is the pure LEVEL z of rv_300s with a "
        "100-row window and NO velocity component; here the same base "
        "(log-transformed) is combined with a 60s crossing test, scoring "
        "only the regime-transition EVENT. The log transform is the "
        "economic input change versus rv_z_300s (which uses raw "
        "rv_300s) -- it re-ranks z by proportional variance distance."
    ),
    info_set="rv_300s",
    inspiration=(
        "iter-003 R5-A family brief: apply the crossing template to "
        "log(rv_300s) with a 60s velocity; the slow base with 60s "
        "crossing targets the 300-900s horizon where round-1/2 found "
        "slow regime state is strongest."
    ),
    compute=compute,
)
