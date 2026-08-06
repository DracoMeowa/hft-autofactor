"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

slope_x_ti_absorb_300s: interaction -- regime-adjusted book thickness x
regime-adjusted aggressive trade imbalance. Thick books absorb aggression.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(book_slope, 300s) x z(trade_imbalance_60s, 300s); warm-up null."""
    slope_z = _z(pl.col("book_slope"), W)
    ti_z = _z(pl.col("trade_imbalance_60s"), W)
    return part.select((slope_z * ti_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="slope_x_ti_absorb_300s",
    mechanism=(
        "Thickness-conditioned aggression: book_slope (depth-accumulation "
        "steepness over both sides) is the book's capacity to ABSORB "
        "marketable flow without price moving. The same aggressive trade "
        "imbalance has different consequences in different shape regimes: "
        "into an unusually THICK book (high slope z -- depth piling up "
        "behind the touch) buy aggression is absorbed by replenished "
        "inventory and its impact decays (fade), while into an unusually "
        "THIN book the identical aggression walks the book and persists "
        "(continuation). The product slope-z x aggression-z therefore "
        "separates absorbed aggression (high x high same sign) from "
        "impact-amplified aggression (low slope x high aggression), a "
        "conditional statement neither leg can make alone. The economic "
        "INPUT is aggressive EXECUTED volume (trade_imbalance_60s), which "
        "distinguishes this from the dead queue_pressure_x_slope (that "
        "used the top-book REBUILD delta as the flow leg and died on "
        "retention); conditioning flow by the shape regime is the "
        "interaction dimension the archive flags as underexploited."
    ),
    info_set="book_slope, trade_imbalance_60s",
    inspiration=(
        "iter-003 R4-D brief direction (b) slope x flow interactions "
        "(steep book absorbing aggressive flow); queue-reactive impact "
        "(Cont-Stoikov-Talreja 2010) conditioned on the depth profile; "
        "deliberately changes the flow leg vs the dead "
        "queue_pressure_x_slope."
    ),
    compute=compute,
)
