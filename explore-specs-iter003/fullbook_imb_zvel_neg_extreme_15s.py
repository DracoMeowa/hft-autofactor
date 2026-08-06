"""Explore-lane prototype spec (iter-003 R6C, family R6C).

fullbook_imb_zvel_neg_extreme_15s: SIGNED-extreme gated z-velocity on the
full-book imbalance. The 15s z-velocity of z_300(full-book imbalance)
weighted by the SIGNED stretch (dz * z), scored ONLY when z < -2.0
(ask-heavy short-stretch). Isolates the short side of the WIDE book --
where hidden-depth ask crowding may signal redemption-flow pressure
invisible at top-5.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 300s trailing z window
LAG = 5  # 15s velocity lookback
GATE = -2.0  # short-stretch gate: z < -2.0


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (sum); null when denominator is 0."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """dz * z when z < -2.0 (short-stretch), else 0.0; warm-up rows null.

    The signed-short gate fires ONLY when the full-book imbalance regime
    is stretched ask-heavy across the ENTIRE visible + hidden depth.
    """
    z = _z(_fullbook_imb(), W)
    dz = z - z.shift(LAG)
    return part.select(
        pl.when(z.is_null() | dz.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when(z < GATE)
        .then(dz * z)
        .otherwise(pl.lit(0.0))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="fullbook_imb_zvel_neg_extreme_15s",
    mechanism=(
        "Short-stretch-isolated broad-book velocity: the 15s z-velocity of "
        "z_300(full-book imbalance) weighted by the SIGNED stretch (dz * "
        "z), but scored ONLY when the regime is stretched ask-heavy "
        "(z < -2.0), zeroed otherwise. The full-book imbalance below "
        "-2sigma means the ENTIRE visible + hidden depth (beyond level 5) "
        "is crowded with ASK liquidity -- large passive sell-side "
        "placement across the full order book, a state that often "
        "accompanies institutional DISTRIBUTION or redemption-flow "
        "pressure invisible to the top-5 engines. Velocity of that "
        "short-stretched broad regime captures the rate at which the "
        "ask-heavy hidden depth is being added or withdrawn -- a "
        "broad-book selling commitment signal that continues at 15-60s. "
        "The signed weighting amplifies by the negative stretch, so the "
        "factor's sign differs from symmetric |z| forms. Distinct from "
        "the oir and wdi neg-stretch variants: the full book captures "
        "depth BEYOND level 5 where redemption-driven passive selling "
        "resides, a different economic source of directional asymmetry "
        "than the touch or top-5 stack."
    ),
    info_set="total_bid_vol, total_ask_vol (batch-2)",
    inspiration=(
        "iter-003 R6C family brief: signed-extreme gating tests "
        "directional asymmetry; this spec isolates the short-stretch "
        "(ask-heavy) tail of the full-book imbalance -- the hidden-depth "
        "base where redemption-flow asymmetry is strongest."
    ),
    compute=compute,
)
