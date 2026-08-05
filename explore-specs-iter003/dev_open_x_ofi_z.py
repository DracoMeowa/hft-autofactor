"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

dev_open_x_ofi_z: deviation from the open anchor gated by sustained
book-flow direction -- does the stretch keep going while informed passive
flow still agrees with it?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_OFI = 100  # 100 x 3s rows = 300s z window on order-book-delta flow


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(mid-open)/open [bps] x z(ofi_60s, 300s); warm-up rows null."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    ofi_z = _z(pl.col("ofi_60s"), W_OFI)
    return part.select((dev * ofi_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="dev_open_x_ofi_z",
    mechanism=(
        "Flow-conditional reversion of the open anchor. dev_from_open_bps "
        "passed all five horizons with NEGATIVE IC (stretch -> reversion) "
        "on average, but reversion is not unconditional: it needs someone "
        "to quote against the stretch. Hypothesis: while sustained "
        "book-flow direction (300s z of ofi_60s) still AGREES with the "
        "stretch, informed passive positioning endorses the move and the "
        "deviation CONTINUES; when flow flips against the stretch the "
        "reversion triggers. The product dev x z(ofi) thus carries "
        "POSITIVE IC: positive product (stretched up with unusual bid "
        "book-building, or down with ask building) -> continuation; "
        "negative product -> accelerated snap-back. OFI is the passive/"
        "limit channel, which round 1 identified as stealth informed "
        "positioning (ofi_z_x_spread_z passed while bare spread died), so "
        "agreement here measures informed endorsement, not chasing. "
        "Falsifiable: a realized negative IC would instead support "
        "'flow agreement exhausts the stretch'; zero IC means flow adds "
        "nothing to the anchor state. The interaction re-ranks stretches "
        "by flow endorsement rather than rescaling the parent."
    ),
    info_set="mid_px, open_px, ofi_60s",
    inspiration=(
        "iter-003 R3-D family brief direction 1 (dev_from_open_bps "
        "conditioned on flow: does reversion only happen when flow "
        "agrees); round-2 all-horizon champion dev_from_open_bps x the "
        "round-1 conditioning lesson (ofi_z_x_spread_z: flow only "
        "predicts under the right state)."
    ),
    compute=compute,
)
