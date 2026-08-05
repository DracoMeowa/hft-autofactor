"""Explore-lane prototype spec (iter-003 R2, slow-currents family R2-D).

ofi_z_x_intens_regime: book-flow z gated by the SLOW activity regime --
OFI unusual vs trailing 300s, multiplied by how hot the 60s-smoothed
event-intensity regime is relative to ITS trailing 300s.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_SMOOTH = 20  # 20 x 3s rows = 60s smoothing of the intensity regime
W_Z = 100      # 100 x 3s rows = 300s z reference window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 300s) x z(rolling_mean(book_event_intensity_60s, 60s), 300s).

    Warm-up (~120 rows) null; constant windows map the gate to 0.0 so the
    product scores neutral, never a spurious signal.
    """
    ofi_z = _z(pl.col("ofi_60s"), W_Z)
    regime = pl.col("book_event_intensity_60s").rolling_mean(
        window_size=W_SMOOTH, min_samples=W_SMOOTH
    )
    gate = _z(regime, W_Z)
    return part.select((ofi_z * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_z_x_intens_regime",
    mechanism=(
        "Flow impact is attention-regime dependent: the same unusual "
        "book-building pressure (high |z| of OFI) carries very different "
        "information depending on whether the market is in a hot or cold "
        "participation regime. Under an unusually HOT 60s-smoothed event "
        "intensity (information-arrival episode), passive book building "
        "on one side is most plausibly informed positioning -- queue "
        "accumulation ahead of moves the informed expect -- so OFI "
        "predicts continuation at 60-300s with full weight. Under an "
        "unusually quiet regime, book-flow deviations are mostly "
        "inventory/re-quoting noise and get gated toward zero. The gate "
        "is the SLOW regime z (300s reference on the 60s-smoothed "
        "intensity), distinct from the transient raw-intensity z "
        "(event_intens_z_300s); round-1 showed state-conditioned "
        "interactions pass where every bare level failed "
        "(ofi_z_x_spread_z), and this swaps the spread state for the "
        "activity state."
    ),
    info_set="ofi_60s, book_event_intensity_60s",
    inspiration=(
        "iter-003 R2 family R2-D brief, direction 4 (gated interaction "
        "ofi_z x intensity regime). Conditioning-on-state lesson from "
        "round 1 (ofi_z_x_spread_z, flow_divergence_x_spread_z); OFI "
        "price impact (Cont, Kukanov & Stoikov 2014); Hawkes episode "
        "structure (Bacry et al. 2015)."
    ),
    compute=compute,
)
