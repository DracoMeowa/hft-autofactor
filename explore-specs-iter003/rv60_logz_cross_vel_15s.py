"""Explore-lane prototype spec (iter-003 R5, family R5-A).

rv60_logz_cross_vel_15s: z-level vs instantaneous-velocity divergence on
log(rv_60s), CROSSING form -- the 300s z of log(rv_60s) crossed zero
within the last 15s; value is the z-velocity, only at crossings, else 0.
Volatility regime acceleration events on the fast realized-variance window.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


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
    """(z_now - z_15s_ago) where sign(z) flipped over 15s, else 0.

    z is computed on log(rv_60s) to tame the right-skew of variance.
    Warm-up rows null; non-crossing rows exactly 0; crossing rows carry
    the signed velocity of the vol-regime transition.
    """
    z = _z(_log_rv(pl.col("rv_60s")), W)
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
    name="rv60_logz_cross_vel_15s",
    mechanism=(
        "Short-vol regime acceleration events: the trailing-300s z of "
        "log(rv_60s) crosses zero within 15s. log compresses the heavy "
        "right tail of realized variance so the z-score is not dominated "
        "by a few extreme spikes. A zero-crossing means 60s realized "
        "variance has just transitioned from below its trailing-300s "
        "norm to above (vol accelerating) or vice versa (vol "
        "decelerating). The crossing VELOCITY measures how sharply the "
        "regime is shifting. Vol regime transitions carry directional "
        "information through two channels: (1) the leverage effect "
        "(Black 1976) -- price drops induce vol spikes, so a positive "
        "crossing velocity (vol jumping above norm) is associated with "
        "recent negative returns and further short-horizon drift; (2) "
        "vol-overreaction mean reversion -- extreme vol spikes are "
        "followed by price stabilization. The SIGNED velocity "
        "distinguishes accelerating from decelerating vol, which the "
        "unsigned rv level cannot. Event-sparse (0 off crossings). "
        "DEDUP: library rv_z_300s is the pure LEVEL z of rv_300s (slow "
        "window, state only); here the base is log(rv_60s) (fast "
        "window, log-transformed) and only the crossing EVENT is scored. "
        "Library signed_rv_60s constructs a directionally-signed rv from "
        "scratch (sign(ret)*ret^2); here we z-score the engine's "
        "unsigned rv_60s and extract directional content from its "
        "velocity. Round-1 lesson: unsigned rv levels dead; signed/"
        "relative forms live."
    ),
    info_set="rv_60s",
    inspiration=(
        "iter-003 R5-A family brief: apply the ofi_z_cross_vel_15s "
        "crossing template to log(rv_60s); rv is unsigned but the z-"
        "velocity of log(rv) is signed and captures vol-regime "
        "transitions."
    ),
    compute=compute,
)
