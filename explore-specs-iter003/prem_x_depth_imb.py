"""Explore-lane prototype spec (iter-003, etf-regime lens).

prem_x_depth_imb: z(iopv_premium, 40 rows = 120s) x RAW wdi.  A deliberate
re-run of the dead iter-001 prem_x_wdi with two structural changes: a fast
120s premium reference frame (vs 300s), and RAW book tilt magnitude instead
of z-scored wdi.  Both changes are the point -- see mechanism.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

#: trailing 120s (40 x 3s rows) causal z-score window for the premium leg
Z_WINDOW = 40


def compute(part: pl.DataFrame) -> pl.Series:
    """z(premium over 120s) x raw wdi; warm-up rows null."""
    x = pl.col("iopv_premium")
    mean = x.rolling_mean(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    std = x.rolling_std(window_size=Z_WINDOW, min_samples=Z_WINDOW)
    z = (x - mean) / std
    zp = pl.when(std.is_not_null() & (std == 0.0)).then(pl.lit(0.0)).otherwise(z)
    w = pl.col("wdi")
    val = zp * w
    return part.select(
        pl.when(zp.is_not_null() & w.is_not_null())
        .then(val)
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="prem_x_depth_imb",
    mechanism=(
        "Flow-backed mispricing episodes continue: a premium stretch "
        "co-occurring with one-sided book depth in the SAME direction is a "
        "flow-driven episode with resting liquidity still feeding it, so "
        "price continues in the premium's direction over 15-60s; a stretch "
        "against the book's tilt is exposed to arbitrage (the thin side "
        "gives arb flow room to push price back to IOPV). Two deliberate "
        "departures from the dead iter-001 prem_x_wdi = z(prem,300s) x "
        "z(wdi,300s): (1) the premium frame is 120s, matching execution "
        "horizons -- a 300s frame averages over several arbitrage cycles "
        "and measures regime, not the actionable now; (2) RAW wdi in "
        "[-1,1] keeps the absolute tilt mass, which is what determines "
        "whether arbitrage flow 'has room': z(wdi) discarded exactly that "
        "distinction (a mildly unusual tilt and an extreme one-sided book "
        "score alike). Positive IC expected: same-sign alignment predicts "
        "continuation."
    ),
    info_set="iopv_premium, wdi",
    inspiration=(
        "iter-001 archive: prem_x_wdi IC ~ 0 in its z(prem,300s) x "
        "z(wdi,300s) form; iter-003 etf-regime brief: premium conditioned "
        "on one-sided books (arb flow has room), re-run with a fast frame "
        "and raw tilt magnitude -- the transform/window changes are the "
        "hypothesis, not cosmetic."
    ),
    compute=compute,
)
