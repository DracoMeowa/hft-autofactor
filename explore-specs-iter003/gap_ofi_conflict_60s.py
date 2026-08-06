"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

gap_ofi_conflict_60s: aggressive print vs order-book delta flow in
OPPOSITE directions -- the book is rebuilding against the last trade.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment (588000)
W = 100       # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(gap_ticks, 300s) x (-sign(z(ofi_60s, 300s))); warm-up null."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    gap_z = _z(gap_ticks, W)
    ofi_z = _z(pl.col("ofi_60s"), W)
    return part.select((gap_z * (-ofi_z.sign())).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="gap_ofi_conflict_60s",
    mechanism=(
        "The last trade vs the book's rebuild direction: a print above the "
        "mid says a buyer crossed; but if the order-book delta flow over "
        "the trailing minute is NEGATIVE (bids pulled / asks stacking), "
        "the book is actively rebuilding AGAINST that print -- the "
        "aggressor's liquidity is being replenished on the opposite side, "
        "the absorption signature. The aggressor gap is therefore expected "
        "to CLOSE: the mid drifts against the print direction over 15-60s "
        "(gap reversal). The mirror -- a below-mid print against positive "
        "book flow -- absorbs seller aggression. Multiplying the gap z by "
        "the NEGATIVE sign of the ofi z is large exactly in the conflict "
        "cell and signed by the aggressor direction, with negative IC "
        "expected (high value -> drift against the print). Unlike "
        "micro_ofi_absorb_60s (pressure leg = passive queue stock), the "
        "pressure leg here is the EXECUTED aggressor event, so the two "
        "specs condition different physical objects on the same flow "
        "channel."
    ),
    info_set="last_px, mid_px, ofi_60s",
    inspiration=(
        "iter-003 R4-D brief direction (c)/(d) -- aggressor vs book-flow "
        "conflict; absorption/iceberg detection (Buti & Rindi 2013); "
        "interaction form keeps distance from the admitted raw gap level."
    ),
    compute=compute,
)
