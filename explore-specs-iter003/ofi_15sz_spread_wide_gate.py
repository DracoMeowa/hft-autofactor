"""Explore-lane prototype spec (iter-003 R5-D, spread raw-level wide gate).

ofi_15sz_spread_wide_gate: short-window BOOK-FLOW SURPRISE (ofi_15s z, 120s)
multiplied by the RAW spread level (quoted_spread_ticks, not its z) -- fast
book-building amplified by the absolute cost of quoting.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 40  # 40 x 3s rows = 120s trailing z window (matches ofi_15s_z_120s library)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_15s, 120s) x raw quoted_spread_ticks; warm-up null."""
    base = _z(pl.col("ofi_15s"), W)
    sp = pl.col("quoted_spread_ticks").cast(pl.Float64)
    return part.select((base * sp).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_15sz_spread_wide_gate",
    mechanism=(
        "Fast book-flow surprise amplified by absolute quoting cost: the "
        "120s z of the 15s order-flow imbalance is multiplied by the RAW "
        "quoted_spread_ticks level (not its z). The economic claim is "
        "level-asymmetric: the same burst of book-building flow is a "
        "stronger continuation signal when the spread is WIDE in absolute "
        "terms, because wide quoting means liquidity providers are already "
        "pricing in adverse-selection risk, so whoever still places "
        "directional limit flow under that cost is paying to be informed "
        "-- the wider the spread the higher the implied conviction toll. "
        "This is distinct from the round-3/4 spread-z gates (clip of "
        "regime-normalized spread z): raw spread level varies tick-by-tick "
        "across the whole session and is near-orthogonal to the base "
        "(panel rho 0.02-0.10 per round-4 finding), so the product does "
        "not reduce to a re-scaled base. Distinct from ofi_z_x_spread_z "
        "(60s base x spread-z, already admitted) on two axes: faster "
        "15s base and raw-level weight, not z-z weight. The raw spread "
        "level is a genuine economic input -- absolute execution cost "
        "conditional on the flow surprise."
    ),
    info_set="ofi_15s, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-D brief direction (a): wide_gate fill-in via RAW "
        "spread level (not spread z) for bases without a wide_gate "
        "variant. ofi_15s_z_120s was round-2 admitted but has no spread "
        "interaction yet; round-4 found raw-spread products are near-"
        "orthogonal to their bases."
    ),
    compute=compute,
)
