"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

fbi_mom_wide_gate: full-book imbalance momentum active ONLY under unusually
wide quoting (one-sided clip gate) -- stress-unlocked pressure momentum.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100   # 100 x 3s rows = 300s trailing spread-state window
D = 20    # 20 x 3s rows = 60s momentum window (as in the base)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """fullbook_imb_mom_60s base x clip(z(spread,300s), 0, inf)."""
    base = _fullbook_imb().diff(D)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    gate = sp_z.clip(lower_bound=0.0)
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fbi_mom_wide_gate",
    mechanism=(
        "Stress-UNLOCKED whole-book pressure momentum, one-sided claim: "
        "the 60s shift of the full-book bid/ask imbalance is scored only "
        "while the quoted spread is unusually WIDE vs its trailing 300s, "
        "and is exactly zero otherwise. Under stressed quoting, makers "
        "withdraw rather than lean, so a whole-book imbalance that still "
        "moves decisively one way marks committed positioning that "
        "precedes impact at 15-60s; outside stressed states the same "
        "momentum is assumed noise and switched off -- no claim that it "
        "reverts there (that stronger claim belongs to the product form). "
        "The exact-zero mass outside stress episodes makes the series "
        "structurally different from both the bare base and the product: "
        "it fires only on the informed-positioning subset of rows. Dedup "
        "note: on its nonzero support the value is monotone in the base "
        "times a positive weight, so sibling corr with fbi_mom_x_spread_z "
        "may be material -- the distinct content is the zero-mass regime "
        "selection."
    ),
    info_set="total_bid_vol, total_ask_vol, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: one-sided-gate variant for the "
        "fullbook imbalance momentum base (R2-C admitted, no spread-z "
        "interaction yet); spread-z the only live interaction dimension "
        "after round 3."
    ),
    compute=compute,
)
