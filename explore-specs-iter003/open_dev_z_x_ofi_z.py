"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_dev_z_x_ofi_z: interaction of the deviation-regime z with the
order-flow regime z -- does queue pressure agree with or fight the
current overextension from the open?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s regime window for both z's


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(dev_from_open_bps, 300s) * z(ofi_60s, 300s); warm-up null."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    return part.select((_z(dev, W) * _z(pl.col("ofi_60s"), W)).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="open_dev_z_x_ofi_z",
    mechanism=(
        "Order-flow agreement with the anchor stretch, both measured "
        "against their own recent regimes. Positive product = queue "
        "pressure is unusually one-sided in the SAME direction as the "
        "unusual stretch: the deviation from the open is being actively "
        "pushed further by order-book delta flow -- fresh-information "
        "extension, where continuation is favored and the anchored "
        "reversion has not yet attracted opposing flow. Negative product "
        "= flow FIGHTS the stretch: an overextended price whose queue "
        "pressure is already turning back toward the open -- the "
        "textbook pre-reversion state, where inventory-carrying desks "
        "and open-referenced arbitrageurs start restoring the anchor. "
        "Both inputs are regime-z'd so the interaction isolates unusual-"
        "vs-unusual co-movement rather than level co-movement; OFI leads "
        "price (queue intent, not executed prints), giving the agreement "
        "test a predictive edge over trade-based confirmation."
    ),
    info_set="mid_px, open_px, ofi_60s",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 5 (deviation x "
        "flow alignment) in the order-book-flow lens; state-conditioned "
        "interactions passed where levels died in rounds 1-2 (ofi_z_x_"
        "spread_z, range_pos_x_spread_z)."
    ),
    compute=compute,
)
