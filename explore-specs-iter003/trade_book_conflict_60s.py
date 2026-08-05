"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

trade_book_conflict_60s: persistent DISAGREEMENT between fast aggression
(trade_imbalance_15s sign) and book flow (ofi_60s sign) over the trailing
60s, scored AGAINST the aggressors -- the book is not confirming.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 20  # 20 x 3s rows = 60s conflict window


def compute(part: pl.DataFrame) -> pl.Series:
    """share(sign(ti_15s) != sign(ofi_60s), both nonzero, 60s) x (-ti_15s).

    Conflict share in [0,1]; multiplied by the negative current
    aggression so sustained conflict fades the aggressor direction.
    Warm-up rows null, never zero-filled.
    """
    st = pl.col("trade_imbalance_15s").sign()
    so = pl.col("ofi_60s").sign()
    conflict = (
        pl.when(st.is_null() | so.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((st != so) & (st != 0) & (so != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    share = conflict.rolling_mean(window_size=W, min_samples=W)
    return part.select((share * (-pl.col("trade_imbalance_15s"))).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="trade_book_conflict_60s",
    mechanism=(
        "Aggression without book confirmation is absorption in progress: "
        "when the 15s trade-imbalance sign and the ofi_60s sign point "
        "OPPOSITE ways for most of the trailing minute, marketable "
        "orders are being filled by passive queue rebuilding on the "
        "contrary side -- the classic iceberg/stealth absorption "
        "signature (Buti & Rindi 2013). The aggressors consume liquidity "
        "that is immediately replenished against them, so their price "
        "impact is temporary and the tape stalls or fades AGAINST the "
        "aggression direction at 15-60s. The factor scores conflict "
        "share x (-current aggression): sustained conflict fades the "
        "aggressor side; when the book confirms (conflict near 0) the "
        "factor is silent by construction -- it never fights confirmed "
        "flow, so it cannot degenerate into a bare contrarian trade-"
        "imbalance level. Distinct from the dead ofi_ti_agree_60s "
        "(same-window share on the two 60s columns, unweighted by "
        "direction): here the disagreement share is read off the fast "
        "15s trade column vs the book column and the output carries the "
        "aggression's sign flipped, turning a regime statistic into a "
        "directional fade."
    ),
    info_set="trade_imbalance_15s, ofi_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R3-C brief direction 2 (disagreement as a warning "
        "signal: trade aggressors hitting a book that is not "
        "confirming); absorption/iceberg detection (Buti & Rindi 2013); "
        "flow_divergence family proved the ofi/ti disagreement axis "
        "carries signal -- this probes its fast/slow sign cell."
    ),
    compute=compute,
)
