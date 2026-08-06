"""Explore-lane prototype spec (iter-003 R6D, family R6D).

ofi_60s_zaccel_extreme_15s: extremity-weighted z-ACCELERATION product on
ofi_60s. The 15s acceleration (2nd difference) of z_300(ofi_60s) weighted
by regime extremity |z|. Tests whether curvature intensification of the
minute-flow regime carries short-horizon continuation signal.
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
    """d2z * |z| where d2z = 15s z-acceleration of ofi_60s; warm-up null."""
    z = _z(pl.col("ofi_60s"), W)
    dz = z - z.shift(LAG)
    d2z = dz - dz.shift(LAG)
    return part.select((d2z * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_60s_zaccel_extreme_15s",
    mechanism=(
        "Acceleration-weighted book-flow regime stretch: the 15s "
        "acceleration of z_300(ofi_60s), weighted by how stretched the "
        "regime is (|z|). ofi_60s is the engine's standard minute-window "
        "order-book-delta flow. Its z-acceleration isolates INTENSIFYING "
        "book-building from steady-state flow: when the minute-flow "
        "regime is already extreme (high |z|: persistent one-sided book "
        "pressure beyond the 300s norm) and its rate of change is "
        "accelerating, the passive program is ramping up -- depth being "
        "posted or pulled at increasing speed over the trailing minute. "
        "This is costlier than steady rebuilding and more likely "
        "committed informed flow, whose impact continues at 15-60s. "
        "Economically distinct from ofi_60s_zvel_extreme_15s (velocity x "
        "|z|): that measures steady fast motion; this measures the "
        "CURVATURE, whether the motion is intensifying or decelerating. "
        "An extreme minute-flow regime with high velocity but zero "
        "acceleration (steady sustained burst) scores ~0 here; only "
        "regimes whose pressure rate is itself changing fire. Also "
        "distinct from the admitted ofi_z_accel_sign_15s (z-acceleration "
        "sign without extremity weight): the |z| weight re-ranks "
        "acceleration by the crowdedness of the regime."
    ),
    info_set="ofi_60s",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel z-acceleration "
        "substrate. The zaccel-extreme template was round-5's strongest "
        "construction (oir_zaccel |t| 27.7) but has not been applied to "
        "the primary OFI column. ofi_60s is confirmed on the panel."
    ),
    compute=compute,
)
