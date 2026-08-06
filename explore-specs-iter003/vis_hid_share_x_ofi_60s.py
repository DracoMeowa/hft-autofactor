"""Explore-lane prototype spec (iter-003 R5-D, hidden-depth x trade flow).

vis_hid_share_x_ofi_60s: touch-concentration z x passive book-flow (ofi_60s)
-- aggressive book-building striking a touch-concentrated (hidden-buffer-
thin) book carries through with amplified directional impact.
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


def _visible_share() -> pl.Expr:
    """(depth_bid5 + depth_ask5) / (total_bid_vol + total_ask_vol)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    db = pl.col("depth_bid5").cast(pl.Float64)
    da = pl.col("depth_ask5").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((db + da) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(visible_share, 300s) x ofi_60s; warm-up null."""
    vis_z = _z(_visible_share(), W)
    ofi = pl.col("ofi_60s")
    return part.select((vis_z * ofi).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="vis_hid_share_x_ofi_60s",
    mechanism=(
        "Passive flow on a concentrated touch: the 300s z of the "
        "visible (top-5) share of total depth multiplied by the raw "
        "ofi_60s. When depth concentrates at the touch (visible share "
        "unusually HIGH) the hidden buffer behind the executable tip is "
        "thin, so passive book-building flow (OFI: limit placements "
        "and cancels at the touch) carries through more forcefully -- "
        "the visible layer is the only thing absorbing the flow with "
        "no deep reserve to dampen it. Positive OFI (bid-side "
        "building) under high visible share predicts up-moves at "
        "15-60s; negative predicts down. Conversely, the same OFI "
        "with LOW visible share (depth parked deep, hidden-heavy book) "
        "is buffered by the hidden reservoir and carries less "
        "information. Distinct from the dead vis_share_x_ti (round-3 "
        "rejected, which used the ACTIVE trade channel trade_imbalance "
        "instead of the PASSIVE limit channel): OFI is the leading "
        "microstructure flow variable (Cont-Kukanov-Stoikov 2014: OFI "
        "dominates signed volume for short-horizon prediction), and "
        "the passive-channel interaction with book structure tests "
        "whether LIMIT-ORDER flow is what the concentrated touch "
        "amplifies. Product of a sign-blind shape z and a signed flow, "
        "structurally different from both parents."
    ),
    info_set=(
        "depth_bid5, depth_ask5, total_bid_vol, total_ask_vol, ofi_60s"
    ),
    inspiration=(
        "iter-003 R5-D brief direction (b): visible-vs-hidden share x "
        "trade pressure. Uses OFI (passive channel) instead of TI "
        "(active channel) because vis_share_x_ti died in round 3 and "
        "OFI is the leading flow predictor per CKS 2014."
    ),
    compute=compute,
)
