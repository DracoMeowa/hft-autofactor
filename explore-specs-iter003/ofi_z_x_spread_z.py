"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ofi_z_x_spread_z: book-building (limit-side) flow conditioned on stressed
quoting -- the passive channel's counterpart of ti_z_x_spread_z.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_OFI = 60      # 60 x 3s rows = 180s z window on order-book-delta flow
W_SPREAD = 100  # 100 x 3s rows = 300s z window on the spread state


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 180s) x z(quoted_spread_ticks, 300s)."""
    ofi_z = _z(pl.col("ofi_60s"), W_OFI)
    sp_z = _z(pl.col("quoted_spread_ticks"), W_SPREAD)
    return part.select((ofi_z * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_z_x_spread_z",
    mechanism=(
        "Stealth book-building under stress: OFI is dominated by limit-flow "
        "at the touch (placement/pull), and its predictive power exceeds "
        "signed trade volume (Cont-Kukanov-Stoikov 2014). When spreads are "
        "stressed wide, opportunistic liquidity providers retreat, so "
        "whoever is still BUILDING the book against the wide quote is "
        "likely informed -- passive queue accumulation precisely when "
        "passive quoting is most dangerous. Bid-side book growth under "
        "wide spreads predicts up-moves at 15-60s; the mirror for asks. "
        "Distinct from ti_z_x_spread_z: this conditions the PASSIVE/"
        "limit-order channel, which carries different (often leading) "
        "information than executed aggression."
    ),
    info_set="ofi_60s, quoted_spread_ticks (library)",
    inspiration=(
        "iter-003 family brief seed 3 (book-flow under stressed quoting); "
        "Cont, Kukanov & Stoikov (2014) 'The price impact of order book "
        "events' -- OFI dominates signed volume; spread-state conditioning "
        "(Stoll 2003)."
    ),
    compute=compute,
)
