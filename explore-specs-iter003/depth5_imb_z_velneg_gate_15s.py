"""Explore-lane prototype spec (iter-003 R5, family R5-B).

depth5_imb_z_velneg_gate_15s: NEW construction -- velocity-SIGN-gated z
on depth5_imb, DECAY side. The z-level of the flat-weighted top-5 depth
imbalance scored ONLY when its own 15s velocity is negative (regime
actively decaying), zeroed otherwise. Tests whether a stretched
visible-depth level mean-reverts specifically during active abandonment.
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


def _depth5_imb() -> pl.Expr:
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    den = db + da
    return (
        pl.when(den > 0.0)
        .then((db - da) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z when dz < 0, else 0.0; warm-up rows null.

    The decay gate (dz < 0) selects rows where the z-regime is actively
    decreasing. During build-up or stagnation (dz >= 0), the output is
    exactly 0. Warm-up is null.
    """
    z = _z(_depth5_imb(), W)
    dz = z - z.shift(LAG)
    return part.select(
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(dz < 0.0)
        .then(z)
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="depth5_imb_z_velneg_gate_15s",
    mechanism=(
        "Decay-isolated visible-depth level: z_300(depth5_imb) scored only "
        "when its own 15s z-velocity is negative (regime actively fading), "
        "zeroed otherwise. The depth5_imb base is the flat-weighted "
        "(non-exp-decayed) top-5 imbalance, proportionally more sensitive "
        "to level 3-5 queue fills than the engine wdi. When this visible-"
        "stack regime is stretched (high |z|) and its velocity turns "
        "negative, the crowded queue is being ABANDONED -- resting orders "
        "pulled across multiple levels -- and the stretched level reverts "
        "at 15-60s as the phantom depth disappears. The one-sided gate "
        "(dz < 0 only) makes NO claim during build-up, where the level "
        "may continue. This is the mirror image of oir_z_velpos_gate_15s: "
        "decay-side rather than build-up-side, and on the broader visible "
        "depth base. Distinct from library depth5_imb_zvel_extreme_15s "
        "(round-4 product form): the gate is a binary velocity-sign "
        "filter, not a magnitude-weighted product."
    ),
    info_set="depth_bid5, depth_ask5",
    inspiration=(
        "iter-003 R5-B family brief: velocity-sign-gated z construction, "
        "DECAY side, on the flat-weighted depth base; tests whether "
        "abandonment of crowded visible depth predicts reversion."
    ),
    compute=compute,
)
