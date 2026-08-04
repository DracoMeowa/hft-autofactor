"""ETF-structure candidate 4: volatility-adjusted book slope (causal residual).

book_slope on 588000 correlates with recent volatility (digest: rho~0.60 vs
rv_300s) -- a steep book partly just encodes that vol has been low.  This
prototype strips that vol component with a CAUSAL trailing-window regression
and keeps only the slope not explained by recent realized vol, isolating the
pure book-shape (support/resistance concentration) information that carries
minutes-scale predictive content.
"""
import polars as pl

#: trailing 300s (100 x 3s rows) causal regression window
REG_WINDOW = 100


def _compute(part: pl.DataFrame) -> pl.Series:
    """Residual of book_slope on rv_300s from a trailing-window OLS.

    beta_t = Cov_trailing(slope, vol) / Var_trailing(vol); the factor is
    slope_t - beta_t * vol_t.  Everything is backward-looking (rolling
    moments), so warm-up and degenerate-variance windows are null.
    """
    x = pl.col("book_slope")
    y = pl.col("rv_300s")
    mx = x.rolling_mean(window_size=REG_WINDOW, min_samples=REG_WINDOW)
    my = y.rolling_mean(window_size=REG_WINDOW, min_samples=REG_WINDOW)
    mxy = (x * y).rolling_mean(window_size=REG_WINDOW, min_samples=REG_WINDOW)
    myy = (y * y).rolling_mean(window_size=REG_WINDOW, min_samples=REG_WINDOW)
    cov = mxy - mx * my
    var = myy - my * my
    beta = cov / var
    resid = x - beta * y
    valid = var.is_not_null() & (var > 1e-12) & resid.is_not_null()
    return part.select(
        pl.when(valid)
        .then(resid)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = {
    "name": "vol_adj_slope",
    "mechanism": (
        "Volatility-adjusted book slope: a steep order book partly just "
        "encodes that recent volatility has been low (book_slope corr "
        "~0.60 with rv_300s). A causal trailing-window regression removes "
        "the vol-explained component, leaving the slope deviation NOT "
        "explained by recent realized vol -- the pure book-shape / "
        "support-resistance concentration signal that carries minutes-scale "
        "(300s+) predictive content, targeting the open 300s gap."
    ),
    "info_set": "book_slope (library), rv_300s (library)",
    "inspiration": (
        "digest: 'book_slope correlated with rv_300s (rho=0.60): on 588000 "
        "slope partly just encodes recent volatility; a vol-adjusted slope "
        "is an obvious mutation' + '300s horizon = weakest and most open'."
    ),
    "compute": _compute,
}
