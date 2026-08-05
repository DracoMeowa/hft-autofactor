"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ofi_concord_15_60: cross-window CONSISTENCY of book flow -- strength of OFI
when the 15s and 60s windows point the SAME direction (sign x min magnitude),
0 when they disagree. Persistence/conviction filter.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """sign(ofi_15s) x min(|ofi_15s|,|ofi_60s|) if same sign else 0.

    Uses the identity min(|a|,|b|) = (|a|+|b|-|a-b|)/2, which collapses to
    0 automatically when a and b have opposite signs, so the expression is
    a single backward-looking combination (warm-up nulls propagate).
    """
    a = pl.col("ofi_15s")
    b = pl.col("ofi_60s")
    mag = (a.abs() + b.abs() - (a - b).abs()) / 2.0
    return part.select((a.sign() * mag).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_concord_15_60",
    mechanism=(
        "Cross-window book-flow conviction: when the 15s and 60s OFI windows "
        "point the same direction, book building is consistent across "
        "timescales -- a sustained one-sided limit-flow investment rather "
        "than a flicker. The factor scores sign x min(|ofi_15s|,|ofi_60s|): "
        "the min caps strength at the WEAKER leg, so a strong fresh impulse "
        "contradicted (or unsupported) by the minute context scores low, and "
        "opposite signs score exactly 0. Persistent same-sign OFI across "
        "windows is the queue-investment signature that reliably precedes "
        "15-30s continuation (OFI predictive power is monotone in its "
        "persistence, CKS 2014), whereas single-window flow flips constantly "
        "and is mostly noise. This is a PERSISTENCE/conviction filter, "
        "economically distinct from the acceleration factors (which measure "
        "the fast-minus-slow gap) -- here agreement itself, capped at the "
        "weaker leg, is the signal."
    ),
    info_set="ofi_15s, ofi_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R2-B brief direction 6 (cross-window same-sign strength); "
        "ofi_15s materialized 2026-08-06; OFI persistence (Cont-Kukanov-"
        "Stoikov 2014); sign-agreement idea complementary to the rejected "
        "ofi_ti_agree_60s (that crossed CHANNELS; this crosses WINDOWS)."
    ),
    compute=compute,
)
