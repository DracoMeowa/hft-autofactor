"""Explore-lane prototype spec (iter-003 R6A, family R6A).

ltns_zaccel_extreme_15s: z-ACCELERATION-extremeness product on the signed
net share of large trades. The 15s acceleration (2nd difference) of the
300s z-regime of large_trade_net_share_60s, weighted by the regime's level
extremity |z|. Mirrors the round-5 winning oir_zaccel template on the
institutional-footprint base.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s lookback for velocity and acceleration


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| where d2z = 15s acceleration of the large-trade-share z.

    Warm-up rows null (z warm-up propagates through two shifts).
    """
    z = _z(pl.col("large_trade_net_share_60s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ltns_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted institutional-footprint stretch: the 15s "
        "acceleration (2nd difference) of z_300(large_trade_net_share_60s), "
        "weighted by how stretched that regime is (|z|). ltns carries the "
        "SIGNED direction of the largest ~10% of prints -- the whale "
        "footprint. Its z-acceleration isolates INTENSIFYING institutional "
        "execution from steady-state large-flow: when the whale regime is "
        "already stretched (high |z|: persistent one-sided large trading "
        "beyond the 300s norm, characteristic of a committed VWAP/TWAP/"
        "implementation-shortfall program) and its curvature is "
        "accelerating further, the program is ramping its child-order "
        "tempo -- the institutional direction continues at 15-60s until "
        "the program completes. Economically distinct from ltns_zvel_div_"
        "60s (round-5, signed-difference of level minus 60s velocity): "
        "that asks overextension-vs-fade over a 60s window; this asks "
        "whether the 15s ACCELERATION (curvature) of the regime is "
        "intensifying. A steady high-velocity whale regime scores ~0 here "
        "(constant dz -> d2z~0); only programs whose execution tempo is "
        "itself changing fire."
    ),
    info_set="large_trade_net_share_60s (batch-2)",
    inspiration=(
        "iter-003 R6A family brief: z-acceleration-extremeness template "
        "applied to the institutional-footprint base NOT yet covered in "
        "acceleration form."
    ),
    compute=compute,
)
