"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ltns_confirms_ti15_z: fast aggression surprise GATED by large-trade
direction -- z_120(trade_imbalance_15s) passes through only when the
signed net share of the largest trades points the same way; otherwise 0.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing z window on trade_imbalance_15s


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(trade_imbalance_15s, 120s) if sign(ltns) agrees, else 0.

    Null z (warm-up) propagates to null output; zero/null ltns blocks the
    gate (scored 0), never zero-fills the warm-up.
    """
    z_ti = _z(pl.col("trade_imbalance_15s"), W)
    ltns = pl.col("large_trade_net_share_60s")
    conf = (
        pl.when(ltns.is_null() | z_ti.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((ltns.sign() == z_ti.sign()) & (ltns != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    return part.select((conf * z_ti).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ltns_confirms_ti15_z",
    mechanism=(
        "Whale-confirmed aggression surprise: the bare fast aggression "
        "surprise z_120(trade_imbalance_15s) died round 2 -- rank-"
        "correlated with panel trade_imbalance_60s, the panel wall. "
        "Gating it through large-trade direction changes BOTH the "
        "economics and the rank structure: an aggression burst whose "
        "sign is corroborated by the net direction of the largest ~10% "
        "of tickets is a program working visibly through marketable "
        "slices WITH big-ticket participation -- the informed-aggressor "
        "state whose information keeps diffusing (Kyle 1985) and "
        "continues at 15-60s. Aggression surprises without whale "
        "corroboration are scored 0 -- likely crowd chasing, stop runs, "
        "or noise. The gate zeroes roughly half the rows, which "
        "structurally breaks the rank correlation with the panel "
        "imbalance level that killed the bare z: the factor now ranks "
        "confirmed surprises against silence instead of ranking "
        "surprise magnitude. Gate form per the R3-C brief (ltns only as "
        "a condition on a flow signal); trade channel, complementing "
        "the book-channel ltns_confirms_ofi_z."
    ),
    info_set="large_trade_net_share_60s, trade_imbalance_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R3-C brief direction 4 (ltns ONLY as a condition/gate "
        "on a flow signal); round-2 death of ti_15s_z_120s (panel "
        "correlation) motivates the confirmation gate as the "
        "decorrelation lever."
    ),
    compute=compute,
)
