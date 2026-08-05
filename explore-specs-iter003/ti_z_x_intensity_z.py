"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ti_z_x_intensity_z: aggressive trade imbalance GATED by feed event
intensity -- aggression is more credible when the book is being rebuilt fast.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_TI = 40   # 40 x 3s rows = 120s z window on trade_imbalance_60s
W_INT = 60  # 60 x 3s rows = 180s z window on event intensity


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(trade_imbalance_60s, 120s) x z(book_event_intensity_60s, 180s)."""
    ti_z = _z(pl.col("trade_imbalance_60s"), W_TI)
    int_z = _z(pl.col("book_event_intensity_60s"), W_INT)
    return part.select((ti_z * int_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ti_z_x_intensity_z",
    mechanism=(
        "Attention-gated aggression: aggressive imbalance arriving while the "
        "order-book event rate is unusually high hits a book that is being "
        "actively re-quoted and defended -- the imbalance is contested and "
        "re-priced within seconds, so its direction is meaningful and "
        "continues at 15-60s. The same imbalance in a quiet tape sits "
        "against a stale book and decays without moving price (or is a "
        "single order, not information). Multiplying z(trade_imbalance_60s) "
        "by z(book_event_intensity_60s) isolates the fast-regime aggression "
        "that precedes continuation. Active-channel mirror of "
        "ofi_z_x_intensity_z; the two can diverge because passive and "
        "aggressive channels respond differently to activity regimes. "
        "Direction from TI, intensity scales conviction."
    ),
    info_set="trade_imbalance_60s, book_event_intensity_60s (library + batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 5 (TI_z x intensity_z attention "
        "gating); book_event_intensity_60s materialized 2026-08-06; "
        "round-1 state-conditioned interaction lesson."
    ),
    compute=compute,
)
