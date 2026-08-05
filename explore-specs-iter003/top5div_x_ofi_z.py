"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

top5div_x_ofi_z: hidden-depth divergence regime timed by sustained book
flow -- the round-2 report's explicit "divergence-flow directional
agreement" follow-up.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing windows


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(wdi - full-book imbalance, 300s) x z(ofi_60s, 300s)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    div_z = _z(pl.col("wdi") - fbi, W)
    ofi_z = _z(pl.col("ofi_60s"), W)
    return part.select((div_z * ofi_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="top5div_x_ofi_z",
    mechanism=(
        "Divergence-flow directional agreement (the round-2 lessons' "
        "explicit follow-up on the strongest-IC factor). The structural "
        "mismatch regime (div_z: displayed touch strength vs deep "
        "backing) predicts best when sustained book FLOW times it: "
        "thin-backed displayed strength (div_z > 0) accompanied by "
        "unusual buy book-building (ofi z > 0) is fragile structure "
        "still being propped up by inbound passive demand -> short-"
        "horizon CONTINUATION while the flow lasts; the same structure "
        "against sell building is exposed and fails faster. Agreement "
        "of structure and flow -> the product div_z x ofi_z carries "
        "POSITIVE IC at 15-60s, with disagreement episodes (negative "
        "product) marking imminent structural failure. Distinct from "
        "the parent (no flow timing -- it averages over flow states), "
        "from dead hidden_imb momenta (raw hidden qty IS-dead; this "
        "stays in the live ratio/z class), and from the dead "
        "iopv_vel_x_ofi_z form (z of a product; this is the product of "
        "two z states)."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol, ofi_60s",
    inspiration=(
        "iter-003 R3-D family brief direction 2 (top5_book_div_z "
        "interactions); round-2 report lesson 2 explicitly suggests "
        "divergence-flow directional agreement as the next lane for the "
        "strongest-IC factor."
    ),
    compute=compute,
)
