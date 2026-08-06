"""Explore-lane prototype spec (iter-003 R5, family R5-C).

oir_zcross_x_ti_sign: touch-queue regime-flip velocity gated by agreement
with trade_imbalance_60s sign. Keeps the oir crossing velocity only when
aggressive trade flow confirms the new direction; zeroes unconfirmed flips.
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


def compute(part: pl.DataFrame) -> pl.Series:
    """oir crossing velocity if sign agrees with ti60, else 0; warm-up null."""
    z = _z(pl.col("oir"), W)
    z_lag = z.shift(LAG)
    flip = (
        pl.when(z.is_null() | z_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((z.sign() != z_lag.sign()) & (z != 0) & (z_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    cross_vel = flip * (z - z_lag)
    ti = pl.col("trade_imbalance_60s")
    agree = (
        pl.when(cross_vel.is_null() | ti.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(cross_vel.sign() == ti.sign())
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
    )
    return part.select((agree * cross_vel).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zcross_x_ti_sign",
    mechanism=(
        "Trade-confirmed touch-queue flips: oir_z_cross_vel_15s scores the "
        "moment control of the best-quote queue changes hands, but a touch "
        "flip can be passive -- one market maker pulls quotes and another "
        "fills the gap with no directional conviction. trade_imbalance_60s "
        "is the executed aggressive-volume balance: positive when buyers "
        "are lifting the ask, negative when sellers hit the bid. "
        "Hypothesis: a touch-queue hand-off whose NEW direction is "
        "corroborated by aggressive trade flow (the crossing velocity and "
        "ti_60s agree in sign) is a genuine informed queue takeover where "
        "both passive and aggressive channels of the same meta-order are "
        "aligned -- and it continues at 300-900s. Unconfirmed flips are "
        "scored exactly 0 (the factor is silent, never contrarian). This "
        "is the GATE form: it asks 'is aggressive-flow confirmation the "
        "necessary condition for touch-flip signal', complementing "
        "wdi_zvel_x_ofi_sign (PRODUCT form, book-flow channel) -- here "
        "the channel is EXECUTED trade flow, not book-delta flow."
    ),
    info_set="oir, trade_imbalance_60s",
    inspiration=(
        "iter-003 R5-C family brief direction 4: condition z-vel winners "
        "on trade pressure sign (direction-isolated); agreement gate on "
        "the oir crossing base."
    ),
    compute=compute,
)
