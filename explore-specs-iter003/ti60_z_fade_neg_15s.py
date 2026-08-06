"""Explore-lane prototype spec (iter-003 R4, family R4-C).

ti60_z_fade_neg_15s: z-level vs instantaneous-velocity divergence on
trade_imbalance_60s, ONE-SIDED form -- the sell-exhaustion quadrant only:
an extreme negative aggression regime whose fast edge has already turned up.
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
    """z where z < 0 and dz > 0 (sell regime already turning up), else 0.

    Warm-up rows null (explicit null mask); all other rows exactly 0
    except the sell-exhaustion quadrant, which carries the level magnitude
    (signed negative).
    """
    z = _z(pl.col("trade_imbalance_60s"), W)
    dz = z - z.shift(LAG)
    sel = (
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z < 0.0) & (dz > 0.0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((sel * z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti60_z_fade_neg_15s",
    mechanism=(
        "Sell-exhaustion quadrant: active only when z_300(trade_"
        "imbalance_60s) is extreme negative (sustained sell aggression "
        "dominating marketable flow) but the 15s z-velocity has already "
        "turned positive -- the selling edge is measurably lifting. Sell "
        "cascades exhaust: marketable sell bursts are typically stop-"
        "driven or program-sliced, and the moment the imbalance itself "
        "starts recovering the forced supply is spent while the price "
        "impact of the cascade is largely temporary (book rebuilds "
        "absorb it), so mid bounces up at 15-60s. Value = the level "
        "magnitude z (signed negative) in the quadrant, exactly 0 "
        "elsewhere: one-sided by design, so it cannot degenerate into a "
        "bare ti level-z or a raw momentum (both dead in earlier "
        "rounds). The sell quadrant is chosen because sell-side cascades "
        "have sharper exhaustion mechanics than buy-side buildups, which "
        "are more often informed accumulation -- the two quadrants are "
        "economically different objects."
    ),
    info_set="trade_imbalance_60s",
    inspiration=(
        "iter-003 R4-C family brief: one-sided quadrant form of the "
        "admitted ofi_z_cross_vel_15s z-vs-velocity template -- the "
        "tension 'slow regime extreme, fast edge already turning' made "
        "event-sparse by quadrant selection."
    ),
    compute=compute,
)
