"""Explore-lane prototype spec (iter-003 R5-D, spread raw-level wide gate).

lastgap_spread_wide_gate: aggressor-side gap (last_px - mid_px in ticks)
multiplied by the RAW spread level -- informed sweep amplified by the
absolute cost-of-quoting it crossed.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

TICK = 0.001  # SSE ETF minimum price increment


def compute(part: pl.DataFrame) -> pl.Series:
    """(last_px - mid_px)/tick x raw quoted_spread_ticks; warm-up null."""
    gap = (pl.col("last_px") - pl.col("mid_px")) / TICK
    sp = pl.col("quoted_spread_ticks").cast(pl.Float64)
    return part.select((gap * sp).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="lastgap_spread_wide_gate",
    mechanism=(
        "Aggressor side amplified by absolute quoting cost: the gap "
        "between the last trade price and the mid (in ticks, signed by "
        "which side was crossed) is multiplied by the RAW "
        "quoted_spread_ticks level. The raw gap already scales with the "
        "half-spread (an aggressive buy at a 2-tick spread lands 1 tick "
        "above mid), so this product captures the JOINT cost of the "
        "sweep: the aggressor not only crossed the half-spread but did "
        "so at a moment when the full spread was elevated -- paying "
        "both the visible toll AND the stress premium. Economically "
        "distinct from a z-gated spread interaction: the raw spread "
        "level directly scales the round-trip cost of acting on the "
        "aggressor signal, so the product encodes the informed-sweep "
        "intensity weighted by how expensive it was to execute. "
        "last_mid_gap_ticks (round-1 admitted, all 5 horizons, nearly "
        "orthogonal to the panel) has no spread interaction yet. The "
        "raw-level product is structurally different from "
        "last_mid_gap_ticks itself (panel rho expected 0.02-0.10 per "
        "round-4 raw-spread finding)."
    ),
    info_set="last_px, mid_px, quoted_spread_ticks",
    inspiration=(
        "iter-003 R5-D brief direction (a): wide_gate fill-in via RAW "
        "spread level. last_mid_gap_ticks was round-1 admitted, the "
        "fastest directional microstructure signal on the panel, with "
        "no spread interaction yet."
    ),
    compute=compute,
)
