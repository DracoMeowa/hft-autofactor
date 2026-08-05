"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ofi_z_x_intensity_z: book-flow imbalance GATED by feed event intensity --
order-flow imbalance is more credible when the book is being rebuilt fast.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_OFI = 40   # 40 x 3s rows = 120s z window on ofi_60s
W_INT = 60   # 60 x 3s rows = 180s z window on event intensity


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 120s) x z(book_event_intensity_60s, 180s)."""
    ofi_z = _z(pl.col("ofi_60s"), W_OFI)
    int_z = _z(pl.col("book_event_intensity_60s"), W_INT)
    return part.select((ofi_z * int_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_z_x_intensity_z",
    mechanism=(
        "Attention-gated book flow: order-flow imbalance measured while the "
        "feed event rate is unusually high is more informative. When "
        "book_event_intensity is elevated the book is being re-quoted, "
        "pulled and hit many times per second -- many participants are "
        "actively revising their quotes, so a concurrent imbalance reflects "
        "broad, deliberate positioning and moves price quickly. The SAME "
        "imbalance during a dead tape may be one stale quote or a single "
        "order and is far less reliable. Multiplying z(ofi_60s) by "
        "z(intensity) up-weights flow signals arriving when the market is "
        "paying attention and suppresses flow noise in low-activity lulls. "
        "Direction comes from OFI; intensity only scales conviction. "
        "Conditions-over-levels per round 1 (spread-z gated OFI passed "
        "while bare spread died)."
    ),
    info_set="ofi_60s, book_event_intensity_60s (library + batch-2 wishlist)",
    inspiration=(
        "iter-003 R2-B brief direction 5 (ofi_z x intensity_z attention "
        "gating); book_event_intensity_60s materialized 2026-08-06; "
        "round-1 state-conditioned interaction lesson."
    ),
    compute=compute,
)
