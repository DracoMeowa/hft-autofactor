"""Explore-lane prototype spec (iter-003 R3, cross-scale flow structure R3-C).

ofi_flip_fade_300s: direction-flip INTENSITY of the book-flow regime --
how often the 300s z of ofi_60s flipped sign over the trailing 300s,
scored AGAINST the current regime (unstable regimes fade).
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s z window AND flip-rate window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """-flip_rate(z-ofi sign, 300s) x z_now; warm-up rows null."""
    z = _z(pl.col("ofi_60s"), W)
    s = z.sign()
    s_lag = s.shift(1)
    flip = (
        pl.when(s.is_null() | s_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((s != s_lag) & (s != 0) & (s_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    rate = flip.rolling_mean(window_size=W, min_samples=W)
    return part.select((-rate * z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_flip_fade_300s",
    mechanism=(
        "Regime instability fades the current leader: count how often "
        "the trailing-300s z of ofi_60s reversed sign over the last 300s "
        "(flip intensity). A HIGH flip rate means the book's flow "
        "regime has been a two-sided battle with no committed "
        "participant -- whichever side happens to lead at this instant "
        "is the side most likely to give way next, because the recent "
        "history says leadership keeps changing hands. The factor "
        "scores -flip_rate x current z: in unstable regimes it leans "
        "AGAINST the current book-flow regime with strength proportional "
        "to both the instability and the regime's current extremity; in "
        "stable regimes (flip rate near 0) it is silent and defers to "
        "the continuation factors. This is a regime-QUALITY statistic "
        "with direction attached -- no registered factor measures the "
        "frequency of OFI-regime reversals; the nearest dead relative "
        "ofi_mom_60s measured z-change magnitude every row, not the "
        "count of sign reversals."
    ),
    info_set="ofi_60s (library)",
    inspiration=(
        "iter-003 R3-C brief direction 5 (intensity of direction flips "
        "of the OFI regime); round-1 meta-lesson that conditions beat "
        "levels -- here the flip-rate conditions a fade of the current "
        "regime; complementary (opposite-support) to the mature-regime "
        "continuation factor."
    ),
    compute=compute,
)
