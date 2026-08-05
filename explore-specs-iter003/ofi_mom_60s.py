"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ofi_mom_60s: 60s momentum (lag-20 delta) of the z-scored book flow --
book-flow acceleration, the delta-over-level play on the OFI channel.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_Z = 60   # 60 x 3s rows = 180s z window on ofi
LAG = 20   # 20 x 3s rows = 60s momentum lag


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 180s) minus its own value 60s ago."""
    z = _z(pl.col("ofi_60s"), W_Z)
    return part.select((z - z.shift(LAG)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_mom_60s",
    mechanism=(
        "Book-flow acceleration: the LEVEL of z-scored OFI says how the "
        "book is positioned; its 60s CHANGE says where limit-flow is "
        "turning. A z-OFI rising over the last minute = passive-side "
        "pressure actively strengthening (queue building speeds up) and "
        "precedes continuation at 15-60s; a falling z-OFI after a bid "
        "build-up flags pull/withdrawal ahead of a down-move. Momentum of "
        "flow is near-orthogonal to flow level by construction. The "
        "delta-over-level meta-lesson (depth momentum wdi_mom_90s lives) "
        "applied to the book-flow channel at a short lag."
    ),
    info_set="ofi_60s (library)",
    inspiration=(
        "iter-003 family brief seed 13; archive meta-lesson: DELTAS/"
        "MOMENTA live where LEVELS died (wdi_mom_90s passed at 900s); "
        "this probes the faster 60s-lag momentum of the OFI channel."
    ),
    compute=compute,
)
