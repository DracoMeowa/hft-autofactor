"""Explore-lane prototype spec (iter-003 R6D, family R6D).

ofi_15s_zvel_extreme_15s: extremity-weighted z-VELOCITY product on the
fastest order-flow imbalance column (ofi_15s). The 15s change rate of
z_300(ofi_15s) weighted by how stretched the regime is (|z|). Applies
the round-4/5 winning zvel-extreme template to the 15s OFI substrate,
which has NOT been put through this construction before.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s = 300s trailing z window
LAG = 5  # 5 x 3s = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 15s z-velocity of ofi_15s; warm-up null."""
    z = _z(pl.col("ofi_15s"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_15s_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted fast book-flow velocity: the 15s change rate "
        "of z_300(ofi_15s), weighted by how extreme the flow regime is "
        "(|z|). ofi_15s resolves order-book-delta flow over quarter-minute "
        "windows -- the freshest passive-channel pressure, where single "
        "placement/cancel bursts still dominate. When this flow regime is "
        "already stretched (high |z|: flow running far beyond the 300s "
        "norm) and its z is MOVING fast (large dz), the burst of book "
        "building or pulling is not just large but rapidly intensifying -- "
        "informed limit flow arriving at increasing speed (Cont-Kukanov-"
        "Stoikov 2014: OFI drives short-horizon price), whose impact "
        "continues at 15-60s. The same velocity around a neutral regime "
        "is routine queue churn and scores near zero because |z| is small. "
        "Economically distinct from the admitted ofi_z_cross_vel_15s "
        "(z sign-flip EVENT scored by velocity, event-sparse): this is an "
        "always-on extremity-weighted velocity product, firing whenever "
        "the flow regime is stretched and moving, not only at crossings. "
        "Also distinct from ofi_15s_z_120s (LEVEL z, no velocity): that "
        "measures the surprise of the level; this measures the SPEED of "
        "the surprise's change, conditioned on the level being extreme."
    ),
    info_set="ofi_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel velocity substrate. "
        "ofi_15s is confirmed on the 59-col panel but has NOT been put "
        "through the zvel-extreme template (only level-z and crossing-vel "
        "constructions exist). Round-4/5 proved extremity-weighted "
        "velocity is the strongest short-horizon template; applying it "
        "to the fastest OFI substrate tests whether sub-15s book-flow "
        "surges carry continuation signal."
    ),
    compute=compute,
)
