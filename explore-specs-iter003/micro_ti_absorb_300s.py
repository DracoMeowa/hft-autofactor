"""Explore-lane prototype spec (iter-003 R4D, quote-shape dynamics).

micro_ti_absorb_300s: PERSISTENT disagreement between top-book microprice
pressure and aggressive trade direction -- duration-weighted absorption.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window (z + occupancy)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """share(sign(microprice_dev) != sign(trade_imbalance_60s), 300s) x
    z(microprice_dev, 300s); warm-up null."""
    sm = pl.col("microprice_dev").sign()
    st = pl.col("trade_imbalance_60s").sign()
    disagree = (
        pl.when(sm.is_null() | st.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((sm != st) & (sm != 0) & (st != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    share = disagree.rolling_mean(window_size=W, min_samples=W)
    micro_z = _z(pl.col("microprice_dev"), W)
    return part.select((share * micro_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="micro_ti_absorb_300s",
    mechanism=(
        "Duration of pressure-vs-aggression conflict: the top-book "
        "microprice pressure sign and the sign of aggressive executed "
        "volume (trade_imbalance_60s) are two independent reads of who is "
        "in control. When they point OPPOSITE ways for most of the trailing "
        "300s, marketable orders have been repeatedly crossing AGAINST the "
        "displayed top-of-book edge -- aggression being filled by passive "
        "replenishment on the other side (absorption). The OCCUPANCY of the "
        "conflict (share of snapshots in disagreement) measures how "
        "committed that absorption regime is, distinct from an "
        "instantaneous snapshot of conflict. Weighting the occupancy by the "
        "current pressure z signs the factor by the pressure direction, so "
        "a high value = sustained upward pressure under persistent sell "
        "aggression -> drift DOWN is expected (and vice versa). Occupancy "
        "statistics are near-orthogonal to the magnitude z-scores (the "
        "duration-vs-intensity separation that made div_pos_frac_300s "
        "live), and this uses the trade channel rather than the ofi "
        "channel (micro_ofi_absorb_60s), so the economic input differs."
    ),
    info_set="microprice_dev, trade_imbalance_60s",
    inspiration=(
        "iter-003 R4-D brief direction (a) micropressure disagreement; "
        "round-3 lesson that occupancy/pos-frac duration forms survive "
        "(div_pos_frac_300s, hidden_imb_pos_frac_300s) while bare levels "
        "die; absorption reading per Buti & Rindi (2013)."
    ),
    compute=compute,
)
