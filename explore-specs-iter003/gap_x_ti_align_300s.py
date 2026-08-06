"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

gap_x_ti_align_300s: interaction -- regime-adjusted aggressor gap x
regime-adjusted aggressive trade imbalance. Price-side and volume-side
aggression pointing the same way = confirmed pressure.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment (588000)
W = 100       # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(gap_ticks, 300s) x z(trade_imbalance_60s, 300s); warm-up null."""
    gap_ticks = (pl.col("last_px") - pl.col("mid_px")) / TICK
    gap_z = _z(gap_ticks, W)
    ti_z = _z(pl.col("trade_imbalance_60s"), W)
    return part.select((gap_z * ti_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="gap_x_ti_align_300s",
    mechanism=(
        "Two-channel confirmation of aggression: the last-mid gap is the "
        "PRICE-side read of who crossed (where the print landed relative "
        "to mid); trade_imbalance_60s is the VOLUME-side read (net "
        "aggressive buy vs sell volume). When both are unusually elevated "
        "in the SAME direction vs their own trailing-300s regimes, two "
        "independent aggression gauges agree -- committed directional "
        "demand/supply whose continuation at 15-60s is well-supported. "
        "When they DISAGREE (price printed up but net aggressive volume "
        "was down), the print is likely a single fill or a stale mark "
        "against the true flow -- the gap is expected to close (reversal "
        "toward the volume side). The product of the two regime z-scores "
        "is positive in the confirmed cell and negative in the conflict "
        "cell; neither leg alone conditions on the other. Both legs are "
        "ratio/z-normalized (raw qty needs ratio/z form), and the "
        "interaction form keeps this away from the admitted raw gap level."
    ),
    info_set="last_px, mid_px, trade_imbalance_60s",
    inspiration=(
        "iter-003 R4-D brief direction (c) gap x side-attributed volume "
        "alignment; condition/interaction meta-lesson (conditions > "
        "levels); Lee & Ready (1991) aggressor classification confirmed "
        "by signed volume."
    ),
    compute=compute,
)
