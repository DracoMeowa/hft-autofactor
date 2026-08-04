"""ETF-structure candidate 2: premium x order-flow interaction.

Premium mean-reversion is enforced by AP creation/redemption arbitrage, but
the SPEED and CONVICTION of that reversion depend on the prevailing order
flow.  When the ETF trades at a premium AND order flow is net selling, the
two forces reinforce reversion; when flow fights the premium, reversion is
delayed.  The product of the (causally z-scored) premium and OFI isolates
this conditional-reversion channel, which the digest names explicitly.
"""
import polars as pl

#: trailing 300s (100 x 3s rows) causal z-score window
Z_WINDOW = 100


def _z(col: str) -> pl.Expr:
    """Causal rolling z-score; warm-up and zero-variance windows are null."""
    x = pl.col(col)
    mean = x.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = x.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    valid = std.is_not_null() & (std > 0.0)
    return (
        pl.when(valid)
        .then((x - mean) / std)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _compute(part: pl.DataFrame) -> pl.Series:
    zp = _z("iopv_premium")
    zf = _z("ofi_60s")
    return part.select(
        pl.when(zp.is_not_null() & zf.is_not_null())
        .then(zp * zf)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = {
    "name": "prem_x_ofi",
    "mechanism": (
        "Premium x order-flow interaction: premium mean-reversion is driven "
        "by AP creation/redemption arbitrage, but its speed depends on the "
        "prevailing order flow. When the ETF is at a premium AND net order "
        "flow is selling, the forces reinforce reversion (stronger predicted "
        "down-move); when flow opposes the premium, reversion is delayed. "
        "The product of the causally z-scored premium and OFI isolates this "
        "conditional-reversion channel rather than either signal alone."
    ),
    "info_set": "iopv_premium (library), ofi_60s (library)",
    "inspiration": (
        "digest: 'premium x order-flow interaction ... orthogonal to the "
        "depth mega-family'; iopv_premium is the strongest mean-reversion "
        "signal (|t|=7.9) and ofi_60s the strongest flow signal, yet their "
        "INTERACTION is unexplored and sits in a different info dimension."
    ),
    "compute": _compute,
}
