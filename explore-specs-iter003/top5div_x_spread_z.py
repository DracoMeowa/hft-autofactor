"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

top5div_x_spread_z: hidden-depth divergence regime (top5_book_div_z_300s
reconstruction) gated by the quoting-stress state -- structural
information is most informative when quoting is toxic.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100         # 100 x 3s rows = 300s trailing windows


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(wdi - full-book imbalance, 300s) x z(quoted_spread_ticks, 300s)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    fbi = (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    div_z = _z(pl.col("wdi") - fbi, W)
    sp_z = _z(pl.col("quoted_spread_ticks"), W)
    return part.select((div_z * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="top5div_x_spread_z",
    mechanism=(
        "Quoting-regime gate on the hidden-depth divergence signal "
        "(top5_book_div_z_300s, the round-2 strongest 15s IC +0.21, "
        "recomputed from panel columns). When spreads are unusually WIDE, "
        "routine two-sided market making retreats and the displayed book "
        "is dominated by whoever deliberately remains: the mismatch "
        "between touch strength and deep backing then carries maximum "
        "adverse-selection information about who is positioning. When "
        "spreads are tight, the same divergence is ordinary recycling "
        "structure with no informational edge. Hypothesis: stress "
        "AMPLIFIES the divergence signal without flipping it -- the "
        "product div_z x spread_z carries POSITIVE IC with a stronger "
        "short-horizon association than the parent, concentrated in "
        "stressed episodes. Falsifiable: IC of the product at or below "
        "the parent's, or a sign flip, rejects the amplification claim. "
        "The spread-z multiplier is near-symmetric and state-only "
        "(spread level is IS-dead), so the product re-ranks divergence "
        "episodes by regime rather than rescaling the parent."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol, quoted_spread_ticks",
    inspiration=(
        "iter-003 R3-D family brief direction 2 (top5_book_div_z "
        "conditioned on spread state); round-2 champion "
        "top5_book_div_z_300s x round-1 conditioning lesson "
        "(ofi_z_x_spread_z)."
    ),
    compute=compute,
)
