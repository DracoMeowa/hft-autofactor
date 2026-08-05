"""Explore-lane prototype spec (iter-003, flow-interaction lens).

flow_accel_signed: z-scored magnitude of flow CHANGE, gated by the current
direction of the flow -- fast-moving tape in a known direction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s z window on |diff| of imbalance


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(|diff(trade_imbalance_60s, 1)|, 180s) x sign(trade_imbalance_60s)."""
    ti = pl.col("trade_imbalance_60s")
    speed = _z(ti.diff(1).abs(), W)
    direction = ti.sign()
    return part.select((speed * direction).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="flow_accel_signed",
    mechanism=(
        "Signed flow-change magnitude: |diff(trade_imbalance)| measures "
        "how FAST aggressive flow is re-pricing, regardless of which way; "
        "its z-score flags episodes where imbalance is swinging hard vs "
        "the 180s regime. Gated by the current imbalance sign, high "
        "values mark episodes where the PREVAILING side is actively "
        "being reinforced or contested in rapid sweeps -- the fast tape "
        "of cascade/sweep events that precede short-horizon continuation "
        "(queue depletion cascades). Distinct functional form from the "
        "linear fast-slow crossovers (ti_ewm_accel_120s, ofi_fast_slow): "
        "the absolute value half-rectifies the change, so this responds "
        "to change MAGNITUDE, not change sign."
    ),
    info_set="trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 16; aggressor-burst/sweep dynamics "
        "(Cartea-Jaimungal-Penalva 2015 ch.8); the change-over-level "
        "meta-lesson with a rectified-magnitude twist to stay orthogonal "
        "to the registered linear acceleration factors."
    ),
    compute=compute,
)
