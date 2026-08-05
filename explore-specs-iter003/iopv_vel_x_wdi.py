"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

iopv_vel_x_wdi: arbitrage-pressure velocity gated by book one-sidedness
-- the ETF only tracks a moving anchor when the book lets it through.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window for the velocity z


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(iopv_velocity, 300s) x wdi; warm-up rows null."""
    vel_z = _z(pl.col("iopv_velocity"), W)
    return part.select((vel_z * pl.col("wdi")).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_vel_x_wdi",
    mechanism=(
        "Book-state gate on arbitrage-pressure velocity. iopv_vel_z_300s "
        "(admitted at 15s) marks unusual NAV re-pricing, but the ETF mid "
        "only TRACKS the moving anchor when the book lets it through: a "
        "one-sided book LEANING WITH the velocity (bid-heavy while the "
        "anchor rises) provides the liquidity path for creation/"
        "redemption execution, so the tracking completes -> continuation "
        "in the velocity direction; a book leaning AGAINST the velocity "
        "is a liquidity wall and the arb stalls until the book turns; a "
        "balanced book (wdi near 0) discounts the velocity entirely. "
        "The product z(vel) x wdi therefore carries POSITIVE IC, and the "
        "wdi multiplier doubles as a one-sidedness magnitude gate -- "
        "exactly the brief's 'IOPV velocity when the book is one-sided "
        "vs balanced'. Structurally distinct from the round-2-dead "
        "iopv_vel_x_ofi_z (z of the velocity x OFI PRODUCT, flow "
        "channel, IS-dead): this is z(velocity) x the resting-book "
        "state, gated on one-sidedness rather than joint-episode z."
    ),
    info_set="iopv_velocity, wdi",
    inspiration=(
        "iter-003 R3-D family brief direction 5 (iopv_vel_z conditioned "
        "on book state); round-2 admitted iopv_vel_z_300s; round-2 "
        "death map: velocity x OFI product form IS-dead -> switch to "
        "resting-book gate."
    ),
    compute=compute,
)
