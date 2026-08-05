"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ofi_sign_persist_60s: fast book-flow COMMITMENT -- fraction of the trailing
20 rows (60s) where consecutive ofi_15s readings kept the SAME sign, centered
and signed by the current flow direction. A second-order run-structure
statistic, not a level or an accumulation.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 20  # 20 x 3s rows = 60s trailing persistence window


def compute(part: pl.DataFrame) -> pl.Series:
    """(share of same-sign consecutive pairs, 60s - 0.5) x 2 x sign(ofi_15s).

    Null inputs propagate (engine warm-up excluded); rolling_mean with
    min_samples=W keeps warm-up rows null, never zero-filled.
    """
    sgn = pl.col("ofi_15s").sign()
    sgn_lag = sgn.shift(1)
    same = (
        pl.when(sgn.is_null() | sgn_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((sgn == sgn_lag) & (sgn != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    share = same.rolling_mean(window_size=W, min_samples=W)
    return part.select(((share - 0.5) * 2.0 * sgn).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_sign_persist_60s",
    mechanism=(
        "Book-flow commitment at the fastest resolved scale: ofi_15s whose "
        "sign flips every snapshot is market-maker churn -- two-sided queue "
        "replacement with no directional information -- while runs of "
        "same-sign 15s book flow mark genuine queue investment that "
        "persistently leans one side of the book. Cont-Kukanov-Stoikov "
        "(2014) show OFI's price power is monotone in its persistence, so "
        "the fraction of consecutive-snapshot pairs keeping the same sign "
        "over the trailing 60s, signed by the CURRENT direction, scores "
        "how committed the passive side is right now: high positive = "
        "sustained buy-side building -> 15-60s continuation up; near zero "
        "= coin-flip chop with no regime. This is a second-order "
        "run-structure statistic: it ignores flow magnitude entirely, so "
        "it is not another ofi_60s/ofi_15s level or z-surprise, and it is "
        "not the dead pos_frac_300s family either (those measured the "
        "fraction of POSITIVE rows of the 60s columns over 300s; this "
        "measures transition-to-transition sign retention of the 15s "
        "column over 60s -- a tape can have the same positive fraction "
        "but alternating signs, which reads as committed here and "
        "positive-fraction-identical there)."
    ),
    info_set="ofi_15s (batch-2 wishlist)",
    inspiration=(
        "iter-003 R3-C brief direction 1 (flow persistence: signed run "
        "statistics of ofi_15s); CKS (2014) OFI persistence; round-2 "
        "lesson that pos_frac level fractions on 60s columns died -- the "
        "transition-match statistic on the 15s column is the unexplored "
        "persistence object."
    ),
    compute=compute,
)
