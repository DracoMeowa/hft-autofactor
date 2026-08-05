"""Explore-lane prototype spec (iter-003, flow-interaction lens).

signed_arrival_z: trade-arrival intensity z-score gated by the current
direction of aggressive flow -- the signed repair of the direction-free
trade_arrival_burst that died in iter-002.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s arrival-intensity z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(n_trades_60s, 180s) x sign(trade_imbalance_60s)."""
    arr_z = _z(pl.col("n_trades_60s"), W)
    direction = pl.col("trade_imbalance_60s").sign()
    return part.select((arr_z * direction).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="signed_arrival_z",
    mechanism=(
        "Direction-conditioned arrival bursts: trade arrivals cluster "
        "during information episodes (Hawkes self-excitation), but "
        "iter-002 proved the DIRECTION-FREE burst level has IC ~ 0 on "
        "588000 -- intensity alone says 'something is happening' without "
        "saying what. Gating the 180s arrival-intensity z-score by the "
        "sign of the concurrent trade imbalance converts it into a "
        "directional flow: clustered arrivals with net buy aggression = "
        "a herd/informed-arrival episode pointing up, and vice versa. "
        "The meta-lesson (flow works when SIGNED and CONDITIONED) applied "
        "to the arrival channel."
    ),
    info_set="n_trades_60s (wishlist), trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 8; iter-002 post-mortem: "
        "trade_arrival_burst (direction-free) IC ~ 0; Engle & Russell "
        "(1998) ACD / Hawkes (1971) self-excitation, signed variant; "
        "order_arrival_60s is never used (always NaN on SSE)."
    ),
    compute=compute,
)
