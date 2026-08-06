"""Explore-lane prototype spec (iter-003 R6D, family R6D).

ofi_15s_zaccel_extreme_15s: extremity-weighted z-ACCELERATION product on
ofi_15s. The 15s acceleration (2nd difference) of z_300(ofi_15s) weighted
by regime extremity |z|. Tests whether the CURVATURE of the fastest
book-flow regime carries signal beyond its velocity.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG = 5


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """d2z * |z| where d2z = 15s z-acceleration of ofi_15s; warm-up null."""
    z = _z(pl.col("ofi_15s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_15s_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted fast book-flow regime stretch: the 15s "
        "acceleration (2nd difference) of z_300(ofi_15s), weighted by how "
        "stretched that regime is (|z|). ofi_15s captures the freshest "
        "quarter-minute of order-book-delta flow. Its z-acceleration "
        "isolates INTENSIFYING book-building from steady-state flow: when "
        "the flow regime is already extreme (high |z|: one-sided book "
        "pressure far beyond the 300s norm) and its rate of change is "
        "ITSELF accelerating, the passive program is ramping up its "
        "commitment -- posting or pulling depth across levels at "
        "increasing speed, which is costlier than steady rebuilding and "
        "more likely informed. The direction of acceleration continues at "
        "15-60s. Economically distinct from ofi_15s_zvel_extreme_15s "
        "(velocity x |z|): that measures steady fast motion of an extreme "
        "regime; this measures the CURVATURE -- whether the motion is "
        "intensifying or decelerating. An extreme flow regime with high "
        "velocity but zero acceleration (steady burst) scores ~0 here; "
        "only regimes whose burst speed is itself changing fire. Also "
        "distinct from the admitted ofi_z_accel_sign_15s (z-acceleration "
        "signed, no extremity weight): the |z| weight re-ranks the "
        "acceleration by the crowdedness of the regime it moves."
    ),
    info_set="ofi_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel z-acceleration "
        "substrate. ofi_15s confirmed on the panel; the zaccel-extreme "
        "template was the round-5 strongest construction (oir_zaccel "
        "|t| 27.7) but has not been applied to the fastest OFI window."
    ),
    compute=compute,
)
