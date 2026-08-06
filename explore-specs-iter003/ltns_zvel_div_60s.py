"""Explore-lane prototype spec (iter-003 R5, family R5-A).

ltns_zvel_div_60s: z-level vs instantaneous-velocity divergence on
large_trade_net_share_60s, SIGNED-DIFFERENCE form -- the slow
institutional-flow regime z minus its own fast z-velocity, itself
regime-normalized.
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


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ltns, 300s) - z(dz, 300s) where dz = 60s z-velocity.

    Warm-up rows null: z warm-up propagates through dz into the
    velocity's own trailing z. Both terms regime-normalized.
    """
    z_e = _z(pl.col("large_trade_net_share_60s"), W)
    dz_e = z_e - z_e.shift(LAG)
    tmp = part.select(z_e.alias("_z"), dz_e.alias("_dz"))
    tmp = tmp.select(pl.col("_z"), _z(pl.col("_dz"), W).alias("_dzz"))
    return tmp.select((pl.col("_z") - pl.col("_dzz")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ltns_zvel_div_60s",
    mechanism=(
        "Institutional flow overextension vs its own fast edge: "
        "z_300(large_trade_net_share_60s) minus the trailing-300s z of "
        "its own 60s z-velocity. ltns carries the SIGNED direction of "
        "the largest ~10% of prints -- the institutional footprint. "
        "Institutional execution programs (VWAP, TWAP, implementation "
        "shortfall) build and complete over minutes, not seconds. When "
        "the ltns z reads extreme (whale flow running hard in one "
        "direction above its 300s norm) but its 60s velocity is already "
        "normalizing (large positive gap: level stretched, velocity "
        "fading), the program is completing -- the last child orders "
        "are firing and the directional pressure is about to cease, so "
        "price mean-reverts at 60-300s. When velocity leads level (large "
        "negative gap: level moderate but velocity surging), a new "
        "program is igniting and directional drift continues. Both "
        "components z-normalized. DEDUP: library ltns_z_180s is the pure "
        "LEVEL z (no velocity); library ltns_delta_60s is raw delta; "
        "library ltns_confirms_* cross ltns with other signals. Here "
        "ltns z is tensioned against ITS OWN velocity -- a build-vs-"
        "exhaust question unique to this spec."
    ),
    info_set="large_trade_net_share_60s (batch-2)",
    inspiration=(
        "iter-003 R5-A family brief: signed-divergence form of the "
        "z-vs-velocity template on large_trade_net_share_60s with a 60s "
        "velocity; the 60s window captures institutional execution "
        "program timescales."
    ),
    compute=compute,
)
