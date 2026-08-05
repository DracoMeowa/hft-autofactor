"""Explore-lane prototype spec (iter-003 R2, short-window flow family R2-B).

ofi_per_depth_z_300s: book-flow PRESSURE PER UNIT DEPTH -- ofi_60s
normalized by the 5-level depth stock, z-scored over the trailing 300s.
How much strain the book is actually absorbing.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s / (depth_bid5 + depth_ask5), 300s); warm-up rows null."""
    depth = pl.col("depth_bid5") + pl.col("depth_ask5")
    strain = (
        pl.when(depth.is_not_null() & (depth > 0.0))
        .then(pl.col("ofi_60s") / depth)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    mean = strain.rolling_mean(window_size=W, min_samples=W)
    std = strain.rolling_std(window_size=W, min_samples=W)
    z = (strain - mean) / std
    return part.select(
        pl.when(std.is_not_null() & (std == 0.0))
        .then(pl.lit(0.0))
        .otherwise(z)
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ofi_per_depth_z_300s",
    mechanism=(
        "Strain-per-depth: the same raw OFI moves price very differently in "
        "a thin book and a deep book. Dividing ofi_60s by the 5-level depth "
        "stock (depth_bid5 + depth_ask5) converts flow into the strain the "
        "book is actually asked to absorb -- Kyle's-lambda intuition that "
        "price change per unit flow is inversely related to available "
        "depth. Unusually high strain-for-depth means the book is absorbing "
        "flow it cannot comfortably hold, so the pressured side gives way "
        "at 15-60s (positive strain against thin depth -> up-move as asks "
        "lift; negative -> down). The trailing-300s z flags strain regimes "
        "unusual for this instrument-day. Economically distinct from raw "
        "OFI z-factors: deep books mechanically shrink the signal and thin "
        "books amplify it, so this re-weights flow by the book's capacity "
        "to resist it -- a liquidity-normalized pressure no registered "
        "factor provides."
    ),
    info_set="ofi_60s, depth_bid5, depth_ask5 (library + base)",
    inspiration=(
        "iter-003 R2-B brief direction 7 (depth-normalized flow pressure); "
        "Kyle (1985) depth-inverse impact; queue-reactive depth conditioning "
        "(Cont-Stoikov-Talreja 2010)."
    ),
    compute=compute,
)
