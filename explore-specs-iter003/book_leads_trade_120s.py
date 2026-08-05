"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

book_leads_trade_120s: LEAD-LAG structure between channels -- how often the
book-flow regime (lagged z of ofi_60s) has been calling the direction that
fast aggression (trade_imbalance_15s) then shows, signed by the current
book regime.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_Z = 100  # 300s trailing z window on ofi_60s
LAG = 2    # 2 x 3s rows = 6s book-regime lead
W_A = 40   # 40 x 3s rows = 120s agreement window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(share(sign(ti15_now) == sign(z_ofi 6s ago), 120s) - 0.5) x sign(z_ofi_now).

    Warm-up rows null (rolling min_samples and shifted z nulls propagate).
    """
    z_ofi = _z(pl.col("ofi_60s"), W_Z)
    sz_lag = z_ofi.shift(LAG).sign()
    st = pl.col("trade_imbalance_15s").sign()
    match = (
        pl.when(st.is_null() | sz_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((st == sz_lag) & (st != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    share = match.rolling_mean(window_size=W_A, min_samples=W_A)
    return part.select(((share - 0.5) * z_ofi.sign()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="book_leads_trade_120s",
    mechanism=(
        "Who leads whom across channels: when the sign of fast aggression "
        "over the trailing two minutes has consistently MATCHED the sign "
        "of the book-flow regime as it stood 6s earlier, the tape's "
        "temporal structure is limit-led -- the book reprices first and "
        "aggressors chase the move, the signature of informed queue "
        "investment diffusing into executions. In a book-led tape the "
        "queue regime is the driver, so continuation at 15-60s proceeds "
        "in the book regime's direction (the factor is signed by the "
        "current z-ofi regime sign, with the lead-agreement share as "
        "strength). In an aggression-led or leaderless tape the match "
        "share sits near 0.5 and the factor is silent. This is a "
        "TEMPORAL-ORDER statistic -- the economic question 'does book "
        "flow lead trade flow?' is untouched by every registered "
        "factor: ofi_ti_agree_60s and flow_divergence compare the two "
        "channels contemporaneously, and ofi_concord_15_60 compares one "
        "channel across windows. Cross-correlation structure between "
        "passive and active flow is a standard microstructure lead-lag "
        "diagnostic (Hasbrouck 1995 price-discovery decomposition)."
    ),
    info_set="trade_imbalance_15s, ofi_60s (batch-2 wishlist + library)",
    inspiration=(
        "iter-003 R3-C brief direction 2 (cross-scale alignment beyond "
        "existing factors -- the lead-lag cell); Hasbrouck (1995) "
        "trade/quote lead-lag discovery; complements the "
        "contemporaneous ti15_sign_x_ofi_z with a temporal-order test."
    ),
    compute=compute,
)
