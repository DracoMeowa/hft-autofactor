"""Explore-lane prototype spec (iter-003 R5, family R5-C).

fullbook_imb_zcross_x_spread_z: broad-book regime-flip velocity gated by
spread-state stress -- the admitted fullbook_imb_z_cross_vel_15s base
(z of the whole-book bid/ask volume ratio crossed zero within 15s) times
the 300s spread z. Tests whether patient full-book repositioning survives
at longer horizons only when spreads are stressed.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s crossing lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """crossing velocity of fullbook imb * z(spread, 300s); warm-up null."""
    z = _z(_fullbook_imb(), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    cross_vel = flip * (z - z_lag)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    return part.select((cross_vel * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zcross_x_spread_z",
    mechanism=(
        "Spread-stress-gated broad-book regime flips: fullbook_imb_z_cross_"
        "vel_15s (the admitted crossing template on the full-book imbalance) "
        "scores sign-reversal events where the ENTIRE book's bid/ask volume "
        "ratio -- including patient deeper-queue institutional positioning "
        "and ETF creation/redemption inventory -- relocates within 15s. "
        "Such a broad relocation is the most costly book event in the family "
        "and hypothesis: under WIDE, stressed spreads this kind of patient "
        "repositioning is credible as informed broad redeployment (the "
        "institutional book does not move the whole stack unless the view "
        "changed); under tight comfortable spreads the same flip can be "
        "routine liquidity rebalancing. Multiplying by z(spread, 300s) "
        "isolates the informed broad relocations from routine ones, "
        "extending the signal to 300-900s where patient positioning matters "
        "most. The broadest base in the spread-z cluster: wdi covers 5 "
        "levels, oir covers 1 level, this covers the whole book -- each "
        "tests the spread-stress hypothesis on a different scale of the "
        "order book, and only the full-book version addresses institutional "
        "patient repositioning."
    ),
    info_set="total_bid_vol, total_ask_vol, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-C family brief: condition the round-4 z-velocity "
        "winner fullbook_imb_z_cross_vel_15s on spread-z; round-2 "
        "established that full-book aggregates carry patient-positioning "
        "information distinct from top-5 depth."
    ),
    compute=compute,
)
