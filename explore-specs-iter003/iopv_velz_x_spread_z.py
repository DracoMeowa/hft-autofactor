"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

iopv_velz_x_spread_z: IOPV velocity surprise (recomputed iopv_vel_z_300s) x
spread-state z -- arbitrage-pressure shock gated by quoting stress.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window (base and gate)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """iopv_vel_z_300s base x z(quoted_spread_ticks, 300s); warm-up null."""
    base = _z(pl.col("iopv_velocity"), W)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((base * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_velz_x_spread_z",
    mechanism=(
        "Arbitrage-latency gating of NAV-velocity shocks: iopv_vel_z_300s "
        "flags the ONSET of unusual fundamental re-pricing of the ETF's "
        "anchor basket. Whether the ETF mid follows quickly depends on the "
        "cost of running the arbitrage -- and that cost is set by the "
        "quoting regime. When the velocity spike lands while spreads are "
        "unusually WIDE, creation/redemption and basket hedging are most "
        "expensive exactly when the anchor moves, so the tracking gap "
        "persists and the ETF continues in the velocity direction over the "
        "following minutes. When the same spike lands under unusually TIGHT "
        "comfortable quoting, arb capital closes the gap within seconds -- "
        "by the time we observe the z spike it is largely priced in "
        "(product side flips toward short-run snap-back). Thus the product "
        "of the two z's encodes a concrete execution-cost mechanism, not a "
        "generic state interaction; the base alone (round-2 admitted at "
        "15s) averages both regimes together."
    ),
    info_set="iopv_velocity, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: iopv_vel_z_300s (R2-D admitted, 15s "
        "alive) still lacks a spread-z interaction; round-2 lesson that "
        "IOPV dynamics (not levels) are the live lane + round-3 that "
        "spread-z is the only live interaction dimension; arb-execution-"
        "cost mechanism."
    ),
    compute=compute,
)
