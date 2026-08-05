"""Explore-lane prototype spec (iter-003 R2-C, trade-structure lens).

size_z_x_ofi_z: granularity regime surge x OFI regime surge -- large
tickets coinciding with unusually strong directional book building.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(log avg_trade_size_60s, 300s) x z(ofi_60s, 300s); warm-up null."""
    size = pl.col("avg_trade_size_60s")
    x = (
        pl.when(size > 0.0)
        .then(size.log())
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    return part.select((_z(x, W) * _z(pl.col("ofi_60s"), W)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="size_z_x_ofi_z",
    mechanism=(
        "Cross-channel informed confirmation: the size-regime surge "
        "multiplied by the OFI-regime surge, both z-scored vs their "
        "trailing 300s. Unusually large tickets coinciding with unusually "
        "strong DIRECTIONAL book-building flow means the passive and "
        "active channels are simultaneously in informed mode -- iceberg-"
        "style layering accompanying block execution on one side. Because "
        "both legs are magnitude-weighted z-scores, the product sign "
        "carries the direction (buy-side agreement positive, sell-side "
        "negative) and the magnitude carries conviction; two independent "
        "channels agreeing marks high-conviction episodes whose direction "
        "continues at 15-60s. Distinct from ofi_ti_agree_60s (a sign-"
        "agreement share between ofi and trade imbalance) and from the "
        "flow_divergence family (ofi vs ti magnitudes): this one "
        "conditions the trade-GRANULARITY regime on book-flow magnitude."
    ),
    info_set="avg_trade_size_60s, ofi_60s",
    inspiration=(
        "iter-003 R2-C family brief direction 6 (avg_trade_size_z x "
        "ofi_60s z); iceberg layering accompanying blocks (Buti & Rindi "
        "2013); cross-channel magnitude agreement; batch-2 trade-"
        "structure columns as the conditioning source."
    ),
    compute=compute,
)
