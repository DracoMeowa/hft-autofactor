"""Explore-lane prototype spec (iter-003, flow-interaction lens).

size_x_direction: average trade-size surge gated by the current direction of
aggressive flow -- institutional footprint in a known direction.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s size z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(avg_trade_size_60s, 180s) x sign(trade_imbalance_60s)."""
    size_z = _z(pl.col("avg_trade_size_60s"), W)
    direction = pl.col("trade_imbalance_60s").sign()
    return part.select((size_z * direction).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="size_x_direction",
    mechanism=(
        "Directional size surge: a rise in the AVERAGE trade size (vs its "
        "own 180s regime) says larger players are active right now; the "
        "sign of the concurrent trade imbalance says which side they hit. "
        "Large prints arriving with net buy aggression = institutional "
        "footprint buying -- meta-order execution continues over minutes "
        "(square-root impact persistence), so 15-60s continuation is "
        "expected. Large prints with balanced/negative aggression flag "
        "distribution. Distinct from ti_z_x_large_share (which weights "
        "z-scored IMBALANCE by the share of volume in big tickets): this "
        "one gates the size REGIME SURGE by direction."
    ),
    info_set="avg_trade_size_60s (wishlist), trade_imbalance_60s (library)",
    inspiration=(
        "iter-003 family brief seed 9; Zhang & Shen (2018) trade-size "
        "decomposition of impact; Bouchaud et al. (2009) large-trade "
        "impact; iter-002 meta-lesson: size statistics only live when "
        "signed/conditioned."
    ),
    compute=compute,
)
