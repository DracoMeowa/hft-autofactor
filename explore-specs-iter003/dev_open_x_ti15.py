"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

dev_open_x_ti15: deviation from the open anchor gated by FAST aggressive
trade direction -- marketable prints into a stretched price as chasing
exhaustion (the competing channel hypothesis to dev_open_x_ofi_z).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """(mid-open)/open [bps] x trade_imbalance_15s; defined from row 0."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    ti = pl.col("trade_imbalance_15s")
    return part.select((dev * ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="dev_open_x_ti15",
    mechanism=(
        "Chasing-exhaustion gate on the open anchor, carried by the FAST "
        "EXECUTION channel -- the competing hypothesis to the passive-book "
        "gate (dev_open_x_ofi_z). Marketable aggression into a price "
        "already stretched from the open is late arrival: with the move "
        "already extended, lifting offers (dev>0, ti_15s>0) hits a "
        "depleted book where inventory holders and makers use the inbound "
        "liquidity to unload, and the stretch snaps back; the mirror for "
        "panic selling into a down-stretch. The product dev x ti_15s "
        "therefore carries NEGATIVE IC: aggression agreeing with the "
        "stretch marks its exhaustion, aggression against the stretch "
        "marks defense/absorption that extends it. trade_imbalance_15s "
        "enters RAW and bounded (a contemporaneous sign state in [-1,1], "
        "not a regime deviation), so the factor is the deviation ranked "
        "by who is currently crossing the spread against it. The pair of "
        "specs identifies WHICH channel conditions reversion: if this IC "
        "is negative while the OFI-gate IC is positive, passive flow "
        "endorses and aggressive flow exhausts -- a clean channel "
        "split predicted by OFI-dominates-signed-volume (Cont, Kukanov & "
        "Stoikov 2014)."
    ),
    info_set="mid_px, open_px, trade_imbalance_15s",
    inspiration=(
        "iter-003 R3-D family brief direction 1 (deviation x "
        "trade_imbalance_15s sign); round-2 champion dev_from_open_bps; "
        "round-1 lesson that short-horizon momentum must carry book/flow "
        "information -- here flow direction gates a slow anchor state."
    ),
    compute=compute,
)
