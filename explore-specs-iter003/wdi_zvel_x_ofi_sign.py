"""Explore-lane prototype spec (iter-003 R5, family R5-C).

wdi_zvel_x_ofi_sign: depth-imbalance extreme velocity multiplied by the
sign of order-flow imbalance (ofi_60s). Tests whether depth-velocity
concordance with actual book-flow direction predicts longer-horizon
returns (confirmation vs divergence of two book channels).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(dz_wdi * |z_wdi|) * sign(ofi_60s); warm-up null."""
    z = _z(pl.col("wdi"), W)
    dz = z - z.shift(LAG)
    zvel = dz * z.abs()
    ofi_sign = pl.col("ofi_60s").sign()
    return part.select((zvel * ofi_sign).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="wdi_zvel_x_ofi_sign",
    mechanism=(
        "Flow-confirmed depth-imbalance velocity: wdi measures the RESTING "
        "5-level depth imbalance (passive queue state) while ofi_60s "
        "measures the active order-flow imbalance (book-delta flow, the "
        "aggressive-vs-passive signed quantity). These are two independent "
        "channels of the same meta-order: informed traders both BUILD the "
        "resting queue (wdi tilts) and TRADE through it (ofi signs). "
        "Hypothesis: when the depth-imbalance velocity direction AGREES "
        "with ofi sign (both bid or both ask), the resting-book rebuild "
        "is confirmed by active book flow -- genuine informed positioning "
        "that continues at 300-900s. When they DIVERGE (book tilts one "
        "way, flow says the other), the queue rebuild is liquidity "
        "provision being picked off, not information -- and the divergence "
        "itself predicts reversal. The product zvel * sign(ofi) encodes "
        "both: positive when the two book channels concur, negative on "
        "divergence. This is the concordance question, NOT a gate: every "
        "row is scored, just signed by agreement. Distinct from ofi_concord "
        "(cross-WINDOW ofi agreement) -- here it is cross-CHANNEL (resting "
        "depth velocity vs active flow direction)."
    ),
    info_set="wdi, ofi_60s",
    inspiration=(
        "iter-003 R5-C family brief direction 4: condition z-vel winners "
        "on ofi sign (buy-pressure confirmation); product form on the wdi "
        "extreme-velocity base to test book-channel concordance."
    ),
    compute=compute,
)
