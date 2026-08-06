"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

micro_ofi_absorb_60s: top-of-book microprice pressure that is NOT confirmed
by net order-book delta flow -- the hidden-absorption cell.
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
    """z(microprice_dev, 300s) x (-sign(z(ofi_60s, 300s))); warm-up null."""
    micro_z = _z(pl.col("microprice_dev"), W)
    ofi_z = _z(pl.col("ofi_60s"), W)
    return part.select((micro_z * (-ofi_z.sign())).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="micro_ofi_absorb_60s",
    mechanism=(
        "Hidden absorption at the touch: microprice_dev is the quantity-"
        "weighted touch price minus the mid (half_spread x oir), so a "
        "positive z means the top-of-book is pressuring UP. ofi_60s is the "
        "net order-book delta flow: negative ofi means limit interest is "
        "being rebuilt AGAINST the current price edge (bids pulled / asks "
        "stacked). When the microprice pressure points up while the book-"
        "flow delta points down, the displayed upward pressure is being "
        "quietly absorbed by passive interest on the other side -- the "
        "classic iceberg signature -- so the pressure is fragile and the "
        "mid tends to drift DOWN (against the pressure) over 15-60s. The "
        "mirror (down pressure on positive flow) absorbs downward. By "
        "multiplying the pressure z by the NEGATIVE sign of the flow z the "
        "factor is large exactly in the unconfirmed/absorption cell and "
        "signed so high values predict against the pressure direction. "
        "This is a two-channel disagreement object, not a bare microprice "
        "level or momentum (both dead oir-clones), so it carries the flow "
        "channel's incremental info."
    ),
    info_set="microprice_dev, ofi_60s",
    inspiration=(
        "iter-003 R4-D brief direction (a) micropressure disagreement vs "
        "OFI sign (hidden absorption). Buti & Rindi (2013) iceberg/"
        "absorption detection; round-1 finding that bare microprice_dev "
        "level/momentum are oir-clones (rho 0.90-0.996) motivates the "
        "cross-channel interaction form instead."
    ),
    compute=compute,
)
