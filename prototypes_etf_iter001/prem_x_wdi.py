"""ETF-structure candidate 3: premium x depth-imbalance interaction.

A positive premium says the ETF is rich vs fair value (arbitrage wants it
down), but the order book's depth imbalance is the opposing/reinforcing
pressure that decides HOW FAST that reversion lands.  A premium paired with
a bid-heavy book (wdi>0) is being propped up, delaying reversion; a premium
paired with an ask-heavy book is about to snap back.  This conditions the
single most independent signal (premium) on the cost-friendly champion
(wdi), a cross-family mutation the digest rates as high marginal value.
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
    zd = _z("wdi")
    return part.select(
        pl.when(zp.is_not_null() & zd.is_not_null())
        .then(zp * zd)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = {
    "name": "prem_x_wdi",
    "mechanism": (
        "Premium x depth-imbalance interaction: a positive premium means "
        "the ETF is rich vs fair value (arbitrage pressure downward), but "
        "the book's depth imbalance decides how fast reversion lands. A "
        "premium paired with a bid-heavy book (high wdi) is being propped "
        "up and reverts slowly; paired with an ask-heavy book it snaps back "
        "fast. The product of causally z-scored premium and wdi captures "
        "this conditional-reversion timing, a cross-family interaction."
    ),
    "info_set": "iopv_premium (library), wdi (library)",
    "inspiration": (
        "digest: wdi is the cost-friendly champion (slowest decay, "
        "half-life 900s) and iopv_premium the most independent signal "
        "(max |rho|=0.23); cross-family mutation (IOPV x depth) has 'far "
        "higher marginal value than more imbalance variants'."
    ),
    "compute": _compute,
}
