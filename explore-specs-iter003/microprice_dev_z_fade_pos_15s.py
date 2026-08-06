"""Explore-lane prototype spec (iter-003 R4, family R4-C).

microprice_dev_z_fade_pos_15s: z-level vs instantaneous-velocity divergence
on microprice_dev, ONE-SIDED form -- the fading-upside quadrant only: an
extreme positive deviation regime whose fast edge has already turned down.
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
    """z where z > 0 and dz < 0 (upside regime already turning), else 0.

    Warm-up rows null (explicit null mask); all other rows exactly 0
    except the fading-upside quadrant, which carries the level magnitude.
    """
    z = _z(pl.col("microprice_dev"), W)
    dz = z - z.shift(LAG)
    sel = (
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z > 0.0) & (dz < 0.0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((sel * z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="microprice_dev_z_fade_pos_15s",
    mechanism=(
        "Fading-upside micro-imbalance quadrant: the factor is active only "
        "when z_300(microprice_dev) is positive (bid queue heavier than "
        "ask, fair-value lead UP) but the 15s z-velocity has already "
        "turned negative -- the upward queue-pressure edge that pushed "
        "the deviation extreme is measurably dissipating. Microprice "
        "deviation leads mid by construction (it is the queue-weighted "
        "fair value), so once the deviation itself starts contracting, "
        "the temporary upside pressure is spent and mid drifts back down "
        "at 15-60s. Value = the level magnitude z (signed positive) in "
        "the quadrant, exactly 0 elsewhere: one-sided by design so the "
        "rank mass cannot degenerate into a bare always-on level-z (the "
        "dead microprice_dev_z_300s form) or a raw momentum (the dead "
        "microprice_dev_mom_60s form). The upside quadrant is chosen "
        "because buy-side micro-extensions on an ETF are typically "
        "demand-shock driven and snap back, whereas downside extensions "
        "carry creation/redemption information -- the two quadrants are "
        "economically different objects."
    ),
    info_set="microprice_dev",
    inspiration=(
        "iter-003 R4-C family brief: one-sided quadrant form of the "
        "admitted ofi_z_cross_vel_15s z-vs-velocity template -- the "
        "tension 'slow regime extreme, fast edge already turning' made "
        "event-sparse by quadrant selection."
    ),
    compute=compute,
)
