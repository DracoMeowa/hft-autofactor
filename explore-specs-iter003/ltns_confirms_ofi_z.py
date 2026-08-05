"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ltns_confirms_ofi_z: book-flow regime surprise GATED by large-trade
direction -- z_300(ofi_60s) passes through only when the signed net share
of the largest trades points the same way; otherwise 0.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window on ofi_60s


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 300s) if sign(large_trade_net_share_60s) agrees, else 0.

    Null z (warm-up) propagates to null output; zero/null ltns blocks the
    gate (scored 0 x z = 0), never zero-fills the warm-up.
    """
    z_ofi = _z(pl.col("ofi_60s"), W)
    ltns = pl.col("large_trade_net_share_60s")
    conf = (
        pl.when(ltns.is_null() | z_ofi.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((ltns.sign() == z_ofi.sign()) & (ltns != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((conf * z_ofi).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ltns_confirms_ofi_z",
    mechanism=(
        "Institutional confirmation of book flow: large_trade_net_share_"
        "60s is the closest available proxy for institutional order "
        "direction on SSE (the signed net share of the largest ~10% of "
        "trades). Bare ltns died all three round-2 attempts -- the "
        "whale-direction level alone is too noisy to carry signal -- "
        "but as a CONFIRMATION GATE on a flow signal it is exactly the "
        "conditional form the archive says to try. A book-flow regime "
        "surprise (z_300 of ofi_60s) whose direction is corroborated by "
        "the net direction of the biggest tickets marks informed queue "
        "investment accompanied by informed executions -- the two "
        "channels of the same meta-order -- and continues at 30-300s. "
        "Unconfirmed flow surprises are scored exactly 0: the factor is "
        "silent rather than contrarian when whales do not corroborate, "
        "so it cannot degenerate into a bare ofi z. This is the GATE "
        "form, book channel -- distinct from the dead ltns_x_ti_60s, "
        "which multiplied ltns by trade_imbalance_60s (product form, "
        "trade channel)."
    ),
    info_set="large_trade_net_share_60s, ofi_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R3-C brief direction 4 (ltns ONLY as a condition/gate "
        "on a flow signal); round-2 lesson that bare ltns is dead but "
        "conditional forms are the open lane; the dead ltns_x_ti_60s "
        "product motivates the gate form on the book channel instead."
    ),
    compute=compute,
)
