"""Explore-lane prototype spec (iter-003 R6D, family R6D).

n_trades_60s_zvel_extreme_15s: extremity-weighted z-VELOCITY product on
the trade-arrival rate (n_trades_60s). The 15s change rate of
z_300(n_trades_60s) weighted by |z|. Tests whether a sudden acceleration
of trading intensity -- when the arrival rate is already abnormal --
predicts short-horizon directional moves.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100
LAG = 5


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * |z| where dz = 15s z-velocity of n_trades_60s; warm-up null."""
    z = _z(pl.col("n_trades_60s"), W)
    dz = z - z.shift(LAG)
    return part.select((dz * z.abs()).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="n_trades_60s_zvel_extreme_15s",
    mechanism=(
        "Extremity-weighted trade-arrival velocity: the 15s change rate "
        "of z_300(n_trades_60s), weighted by how abnormal the arrival "
        "regime is (|z|). n_trades_60s counts executions in the trailing "
        "minute -- a direct measure of trading intensity (attention and "
        "urgency). When the arrival rate is already extreme (high |z|: "
        "far more trades than the 300s norm -- an active informed "
        "regime) and its z is rising fast (large positive dz), trading "
        "intensity is SURGING further: more participants are hitting "
        "quotes at increasing rate, which precedes directional impact as "
        "the aggressive flow depletes resting liquidity. When dz is "
        "negative and |z| is high, the burst is EXHAUSTING -- the active "
        "regime is cooling, and the price impact that already occurred "
        "tends to partially revert. The |z| weight suppresses velocity "
        "around normal arrival regimes (routine trading). Economically "
        "distinct from the dead trade_count_z_300s and ntrades_pace_z_300s "
        "(LEVEL z-scores of trade count, which round-2 found IS-dead): "
        "those ask 'is the count high?'; this asks 'is the count's rate "
        "of change extreme, AND is the level already abnormal?' -- a "
        "velocity-extremity product, not a level statistic. Also distinct "
        "from trade_imbalance velocity (which is directional): this "
        "measures unsigned arrival-rate dynamics, capturing urgency "
        "irrespective of trade direction."
    ),
    info_set="n_trades_60s (wishlist batch-1)",
    inspiration=(
        "iter-003 R6D family brief direction 2: novel velocity substrate. "
        "n_trades_60s is confirmed on the panel. Round-2 found trade-"
        "count LEVEL z-scores dead (0/6), but the zvel-extreme template "
        "measures the velocity of the z-regime weighted by its "
        "extremity -- a fundamentally different economic question "
        "(urgency dynamics vs level). The template was validated on "
        "book-imbalance substrates; this tests it on the arrival-rate "
        "substrate."
    ),
    compute=compute,
)
