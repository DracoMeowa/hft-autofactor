"""Explore-lane prototype spec (iter-003, etf-regime lens).

vol_rate_x_ti: z(per-snapshot volume increment, 100 rows) x sign of
trade_imbalance_60s.  HEAVY TAPE IN THE FLOW DIRECTION: volume surprise
conditioned on aggressor-side direction (the sibling vol_confirmed_mom
conditions on realized price direction instead).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 300s (100 x 3s rows) z window for volume increments
Z_WINDOW = 100


def compute(part: pl.DataFrame) -> pl.Series:
    """z(volume increment) x sign(trade imbalance); warm-up rows null."""
    vinc = pl.col("cum_trade_vol").diff()
    mean = vinc.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = vinc.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    zv = (vinc - mean) / std
    zv = pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(zv)
    ti = pl.col("trade_imbalance_60s")
    val = zv * ti.sign()
    return part.select(
        pl.when(zv.is_not_null() & ti.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="vol_rate_x_ti",
    mechanism=(
        "Heavy tape in the flow direction = informed flow persistence: "
        "when the trailing-60s aggressive volume balance is one-sided AND "
        "traded volume per snapshot is unusually heavy versus the trailing "
        "300s, the flow is more likely to carry an informed component "
        "(Kyle-style: informed traders trade large AND directionally), so "
        "the price impact persists and the flow direction continues at "
        "30-300s. Heavy-but-BALANCED tape scores ~0 by construction -- it "
        "is churn (inventory reshuffling, two-sided liquidity provision) "
        "and predicts little; light tape also scores ~0 because small "
        "imbalances on thin volume are noise. The direction comes from the "
        "aggressor flow itself rather than from realized price momentum, "
        "which makes this the flow-leading member of the volume-surprise "
        "pair (vol_confirmed_mom uses price-momentum sign)."
    ),
    info_set="cum_trade_vol, trade_imbalance_60s",
    inspiration=(
        "iter-003 etf-regime brief: z of per-snapshot volume increment x "
        "sign(trade_imbalance_60s); Kyle (1985) informed-flow intensity; "
        "cum_trade_vol materialized 2026-08-05 (wishlist); iter-002 "
        "meta-lesson: interactions of fast state survive where levels die."
    ),
    compute=compute,
)
