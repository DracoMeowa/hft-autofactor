"""Explore-lane prototype spec (iter-003 R5-D, spread raw-level wide gate).

ofi_pdepth_wide_walk: OFI strain per unit depth (ofi_60s / 5-level depth,
z-scored over 300s) active ONLY when the raw spread exceeds its rolling
mean -- book-strain that matters exclusively in wide-quoting episodes.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window and spread-mean window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s/depth, 300s) x 1{spread > rolling_mean}; warm-up null."""
    depth = pl.col("depth_bid5") + pl.col("depth_ask5")
    strain = (
        pl.when(depth.is_not_null() & (depth > 0.0))
        .then(pl.col("ofi_60s") / depth)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    base = _z(strain, W)
    sp = pl.col("quoted_spread_ticks").cast(pl.Float64)
    sp_mean = sp.rolling_mean(window_size=W, min_samples=W)
    gate = (
        pl.when(sp_mean.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(sp > sp_mean)
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
    )
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_pdepth_wide_walk",
    mechanism=(
        "Book-strain active ONLY under wide quoting: the 300s z of the "
        "OFI-per-depth strain is zeroed unless the raw quoted spread "
        "exceeds its trailing-300s mean, exactly the binary on/off "
        "complement to a continuous raw-spread weight. The strain "
        "itself (ofi_60s normalized by the 5-level depth stock) measures "
        "how much flow-pressure the book is absorbing per unit of "
        "available depth -- high strain means the book is being asked to "
        "hold flow it cannot comfortably absorb. This claim is "
        "exclusively about the WIDE-spread subset: when the spread "
        "exceeds its mean, makers have stepped back, so the strain "
        "registers against an already-withdrawn book and the pressured "
        "side is more likely to give way; under comfortable sub-mean "
        "spreads the same strain is absorbed by competitive depth and "
        "is switched off. ofi_per_depth_z_300s (round-2 admitted) has "
        "no spread interaction yet. The binary gate form (0 outside the "
        "wide regime) is structurally different from both the base "
        "(continuous everywhere) and a continuous raw-spread weight "
        "(nonzero everywhere) -- it is an episode detector."
    ),
    info_set="ofi_60s, depth_bid5, depth_ask5, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-D brief direction (a): wide_gate fill-in via a "
        "rolling-median threshold gate (binary on/off) for the "
        "ofi_per_depth_z_300s base (round-2 admitted, Kyle-lambda "
        "strain-per-depth, no spread interaction yet)."
    ),
    compute=compute,
)
