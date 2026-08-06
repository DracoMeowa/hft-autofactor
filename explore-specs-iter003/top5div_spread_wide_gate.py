"""Explore-lane prototype spec (iter-003 R5-D, spread raw-level wide gate).

top5div_spread_wide_gate: structural touch-vs-queue mismatch (top5_book_div
z, 300s) multiplied by the RAW spread level (quoted_spread_ticks, not its
z) -- fragile displayed strength amplified by absolute quoting cost.
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


def _fullbook_imb() -> pl.Expr:
    """(total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)."""
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """z(wdi - fullbook_imb, 300s) x raw quoted_spread_ticks; warm-up null."""
    base = _z(pl.col("wdi") - _fullbook_imb(), W)
    sp = pl.col("quoted_spread_ticks").cast(pl.Float64)
    return part.select((base * sp).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="top5div_spread_wide_gate",
    mechanism=(
        "Structural mismatch amplified by absolute quoting cost: the "
        "300s z of the (wdi minus full-book imbalance) gap is multiplied "
        "by the RAW quoted_spread_ticks level. A persistent positive z "
        "marks displayed touch strength that is NOT backed by the deep "
        "queue -- thin-backed strength that is fragile and expected to "
        "drift down. Multiplying by the raw spread level tests whether "
        "this fragility is more acute when quoting is expensive in "
        "absolute terms: the same thin-backed posture at a 2-tick spread "
        "is under more stress than at a 1-tick spread because the cost "
        "to defend it (or to fade it) is larger, so the drift is expected "
        "to be faster. The raw level is near-orthogonal to the base "
        "(round-4 finding), so the product is not a re-scaled base. "
        "Distinct from top5div_x_spread_z (regime-normalized spread-z "
        "gate): the absolute-cost weighting does not center the spread "
        "on its trailing mean, so it carries the level information the "
        "z-gate discards."
    ),
    info_set="wdi, total_bid_vol, total_ask_vol, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-D brief direction (a): wide_gate fill-in via RAW "
        "spread level for top5_book_div_z_300s (round-2 admitted, "
        "strongest short-horizon IC, no raw-spread interaction yet)."
    ),
    compute=compute,
)
