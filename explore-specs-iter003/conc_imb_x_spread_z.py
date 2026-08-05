"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

conc_imb_x_spread_z: placement-urgency regime gated by quoting stress --
head-heavy one-sided concentration under wide spreads is informed
queue-grabbing, not routine market making.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing windows


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(bid/ask concentration asymmetry, 300s) x z(spread_ticks, 300s)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    cb = pl.when(tb > 0.0).then(db / tb).otherwise(pl.lit(None, dtype=pl.Float64))
    ca = pl.when(ta > 0.0).then(da / ta).otherwise(pl.lit(None, dtype=pl.Float64))
    conc_z = _z(cb - ca, W)
    sp_z = _z(pl.col("quoted_spread_ticks"), W)
    return part.select((conc_z * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="conc_imb_x_spread_z",
    mechanism=(
        "Quoting-stress gate on the placement-style regime. Under TIGHT "
        "normal quoting, one side parking orders at the head is routine "
        "market-making posture with weak directional content. Under an "
        "unusually WIDE spread, passive quoting is dangerous -- whoever "
        "still stacks one side's orders at the touch is competing for "
        "queue priority despite adverse selection, i.e. urgent informed "
        "positioning that expects the spread to resolve in its favor. "
        "Hypothesis: stressed quoting AMPLIFIES the concentration "
        "signal: product conc_z x spread_z carries POSITIVE IC at slow "
        "horizons (300-900s, the parent's admitted horizons), "
        "concentrated in stressed episodes. Falsifiable: IC at or below "
        "the parent's rejects the amplification claim. Distinct from "
        "the rv-gate sibling conc_imb_x_rv_z (exogenous turbulence vs "
        "endogenous quoting stress -- different conditioning states, "
        "different economics) and from the parent conc_imb_z_300s "
        "(which averages over the quoting regime)."
    ),
    info_set="depth_bid5, depth_ask5, total_bid_vol, total_ask_vol, quoted_spread_ticks",
    inspiration=(
        "iter-003 R3-D family brief directions 2/6 (book-state x "
        "quoting-state gates); round-2 admitted conc_imb_z_300s; "
        "round-1 spread-as-condition lesson (ofi_z_x_spread_z)."
    ),
    compute=compute,
)
