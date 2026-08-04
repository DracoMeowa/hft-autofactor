"""oir_z_300s — trailing-300s rolling z-score of the library oir.

Smoke prototype for the explore lane: strictly causal (backward-looking
rolling window only), warm-up rows null, zero-variance windows mapped to 0.
"""
from __future__ import annotations

import polars as pl


def compute(part: pl.DataFrame) -> pl.Series:
    """Causal z-score of oir over 100 rows (300s); warm-up (< 100 rows) null."""
    x = pl.col("oir")
    mean = x.rolling_mean(window_size=100, min_samples=100)
    std = x.rolling_std(window_size=100, min_samples=100)
    z = (x - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = dict(
    name="oir_z_300s",
    mechanism=(
        "order-imbalance persistence: top-of-book pressure (oir) unusually "
        "extreme vs its own trailing 300s distribution flags sustained "
        "one-sided demand that should move the mid in the same direction "
        "over the following seconds to minutes."
    ),
    info_set="oir (library)",
    inspiration=(
        "queue-reactive price impact (Cont, Stoikov & Talreja 2010); "
        "z-scored analogue of the library oir, mirroring the built-in "
        "spread_z_300s convention."
    ),
    compute=compute,
)
