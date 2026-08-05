"""Explore-lane prototype spec (iter-003 round 3, state-interaction family R3-D).

dev_open_x_spread_z: deviation from the open anchor gated by stressed
quoting -- liquidity-provider withdrawal at the stretched level
accelerates the snap-back.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_SPREAD = 100  # 100 x 3s rows = 300s z window on the spread state


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(mid-open)/open [bps] x z(quoted_spread_ticks, 300s)."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    sp_z = _z(pl.col("quoted_spread_ticks"), W_SPREAD)
    return part.select((dev * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="dev_open_x_spread_z",
    mechanism=(
        "Quoting-stress gate on the open anchor. A spread unusually wide "
        "for its trailing-300s self WHILE price sits stretched from the "
        "open means liquidity providers are retreating from quoting the "
        "stretched level -- adverse-selection and inventory fear exactly "
        "where the reversion pressure lives. Reversion needs someone to "
        "quote against the stretch; withdrawal removes the cushion, so "
        "the eventual snap-back is faster and sharper. The product dev x "
        "z(spread) carries NEGATIVE IC: stressed-wide quoting amplifies "
        "the reversion of the stretch; tight quoting at a stretched "
        "price is confident absorption and the deviation decays slowly. "
        "Round 1 established spread has no bare directional content "
        "(spread_z_60s/120s IS-dead) yet works as a condition "
        "(ofi_z_x_spread_z, flow_divergence_x_spread_z passed). Distinct "
        "from the admitted range_pos_x_spread_z: different anchor (open "
        "price vs envelope position; the anchors correlate ~0.77 but the "
        "HYPOTHESIS direction differs -- reversion amplification here vs "
        "boundary-resolution there), so the ranking change is not a "
        "rescaling of the admitted interaction."
    ),
    info_set="mid_px, open_px, quoted_spread_ticks",
    inspiration=(
        "iter-003 R3-D family brief direction 6 (two-state gates; vary "
        "the range_pos_x_spread_z question); round-2 champion "
        "dev_from_open_bps x round-1 spread-as-condition lesson (Stoll "
        "2003 on spread state-dependence)."
    ),
    compute=compute,
)
