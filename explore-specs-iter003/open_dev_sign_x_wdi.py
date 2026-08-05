"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_dev_sign_x_wdi: 5-level depth imbalance gated by which side of the
open the price is stretched to -- does resting inventory lean WITH the
anchor stretch or back toward the anchor?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """sign(mid - open) * wdi; defined from the first row."""
    dev_sign = (pl.col("mid_px") - pl.col("open_px")).sign()
    return part.select((dev_sign * pl.col("wdi")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="open_dev_sign_x_wdi",
    mechanism=(
        "Inventory positioning relative to the anchor stretch, measured on "
        "the QUOTE-STOCK side (resting depth, not flow). When mid is "
        "stretched above the open, a bid-heavy 5-level book (positive "
        "wdi) means resting liquidity SUPPORTS the extension from below: "
        "market makers are comfortable long inventory away from their "
        "reference price, the anchored pull is being absorbed, and the "
        "extension tends to persist. An ask-heavy book under an above-"
        "open stretch is the opposite: inventory leaning back toward the "
        "anchor, quotes positioned to fade the move, reversion pressure "
        "stored in the book itself. Gating wdi by the SIGN of the open-"
        "deviation (not its magnitude) makes the factor a pure alignment "
        "test and keeps it bounded [-1,1]. This is the third, independent "
        "flow/book lens on deviation alignment: OFI (queue intent) and "
        "trade imbalance (executed aggression) are FLOWS; depth imbalance "
        "is the STOCK of committed liquidity, which adjusts on slower, "
        "more deliberate timescales and conditions the 60-300s horizons."
    ),
    info_set="mid_px, open_px, wdi",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 5 (deviation x "
        "book/flow alignment) in the depth-stock lens; retry of the "
        "range-position x wdi idea (round-2, dead) on the anchor axis "
        "that won round 2, with sign-gating instead of level-magnitude "
        "gating."
    ),
    compute=compute,
)
