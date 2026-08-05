"""Explore-lane prototype spec (iter-003, flow-interaction lens).

gap_x_direction: quiet-tape state (long gaps between trades) gated by the
direction of recent aggressive flow -- digestion vs exhaustion.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s gap z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(trade_gap_ms, 180s) x sign(trade_imbalance_60s)."""
    gap_z = _z(pl.col("trade_gap_ms"), W)
    direction = pl.col("trade_imbalance_60s").sign()
    return part.select((gap_z * direction).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="gap_x_direction",
    mechanism=(
        "Digestion vs exhaustion after directional flow: an unusually long "
        "gap since the last trade (high z of trade_gap_ms) means the tape "
        "has gone quiet. WHAT the quiet follows determines its meaning: "
        "silence after one-sided buy aggression is either digestion "
        "(inventory absorbed, continuation) or exhaustion (move over, "
        "reversion). Direction-free gap levels cannot distinguish, which "
        "is why arrival/gap LEVELS died in iter-002; gating by the sign "
        "of the concurrent imbalance gives the market-state a direction. "
        "The IC sign then empirically resolves digestion vs exhaustion "
        "for 588000 at 30-300s horizons."
    ),
    info_set="trade_gap_ms (wishlist), trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 10; ACD/inter-trade-duration "
        "literature (Engle & Russell 1998; Lillo, Farmer & Mantegna "
        "2003); iter-002 lesson that direction-free tape-state levels "
        "carry no IC."
    ),
    compute=compute,
)
