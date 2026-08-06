"""Explore-lane prototype spec (iter-003 R4, family R4-C).

spread_zvel_extreme_15s: z-level vs instantaneous-velocity divergence on
quoted_spread_ticks, PRODUCT form -- the 15s z-velocity of the spread
regime weighted by the extremity |z| of the regime being moved.
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


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 15s z-velocity of the spread regime.

    Warm-up rows null (z warm-up propagates through the shift).
    """
    z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="spread_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted velocity of the quoting-cost regime: the 15s "
        "change rate of z_300(spread) weighted by |z|. Spread motion out "
        "of an ALREADY extreme quoting regime is the informative kind: "
        "fast widening off an extreme-wide state is makers capitulating "
        "to adverse selection (fear escalating), which precedes continued "
        "downside pressure at 15-60s, while fast tightening off an "
        "extreme state is liquidity being restored and precedes "
        "stabilization/upside. Around a neutral spread regime the same "
        "velocity is routine tick oscillation and scores ~0 via the "
        "extremity weight. A level-x-velocity interaction with direction "
        "carried by the velocity: different from the built-in "
        "spread_z_300s (level only, IS-dead bare) and from the dead "
        "spread window swaps -- the live dimension per round 1/3 is "
        "conditioned/derivative spread forms, and this is a derivative "
        "one."
    ),
    info_set="quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-C family brief: product form of the admitted "
        "ofi_z_cross_vel_15s z-vs-velocity template applied to the "
        "quoting-cost state column."
    ),
    compute=compute,
)
