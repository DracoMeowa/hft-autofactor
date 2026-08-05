"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ti_z_x_spread_z: signed aggressive flow conditioned on stressed (wide)
quoting -- imbalance arriving when market makers have widened up.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_TI = 60      # 60 x 3s rows = 180s z window on aggressive flow
W_SPREAD = 100  # 100 x 3s rows = 300s z window on the spread state


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(trade_imbalance_60s, 180s) x z(quoted_spread_ticks, 300s)."""
    ti_z = _z(pl.col("trade_imbalance_60s"), W_TI)
    sp_z = _z(pl.col("quoted_spread_ticks"), W_SPREAD)
    return part.select((ti_z * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_z_x_spread_z",
    mechanism=(
        "Stress-conditioned aggressive flow: a quoted spread wide relative "
        "to its own trailing distribution flags dealer retreat and elevated "
        "adverse-selection fear. Signed trade imbalance arriving in that "
        "state has outsized information content -- the few makers left quote "
        "thin, so directional aggression both reveals information and moves "
        "price further per unit volume. Positive product (buy imbalance "
        "under wide spreads) predicts up-continuation at 15-60s; negative "
        "product (buy imbalance under unusually TIGHT spreads) is flow "
        "hitting a deep, competitive book with little follow-through. The "
        "interaction is near-orthogonal to both parent z-scores by sign "
        "symmetry."
    ),
    info_set="trade_imbalance_60s, quoted_spread_ticks (library)",
    inspiration=(
        "iter-003 family brief seed 2 (signed flow conditioned on stressed "
        "quoting); spread-state conditioning of impact (Stoll 2003, "
        "componens of the bid-ask decomposition); meta-lesson that "
        "interactions carry signal where levels died."
    ),
    compute=compute,
)
