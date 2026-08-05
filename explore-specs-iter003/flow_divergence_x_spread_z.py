"""Explore-lane prototype spec (iter-003, flow-interaction lens).

flow_divergence_x_spread_z: the champion's absorption signal conditioned on
spread stress -- divergence is most informative when quoting is stressed.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W_DIV = 40      # 40 x 3s rows = 120s divergence z windows
W_SPREAD = 100  # 100 x 3s rows = 300s spread-state z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """(z(ofi,120s) - z(ti,120s)) x z(quoted_spread_ticks, 300s)."""
    div = _z(pl.col("ofi_60s"), W_DIV) - _z(pl.col("trade_imbalance_60s"), W_DIV)
    sp_z = _z(pl.col("quoted_spread_ticks"), W_SPREAD)
    return part.select((div * sp_z).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="flow_divergence_x_spread_z",
    mechanism=(
        "Stress-gated absorption: the flow_divergence signal (book flow "
        "minus trade flow) is strongest when it must EXPLAIN itself "
        "against stressed quoting. When spreads are wide (makers fearful, "
        "adverse selection high), a book that still builds one side "
        "beyond what executed aggression justifies is almost certainly "
        "informed stealth flow -> strong continuation at 15-60s. The "
        "same divergence under unusually tight, comfortable spreads may "
        "be routine queue noise. Multiplying by spread z gates the "
        "champion signal by the quoting state; the product is sign-"
        "symmetric and near-orthogonal to either parent."
    ),
    info_set="ofi_60s, trade_imbalance_60s, quoted_spread_ticks (library)",
    inspiration=(
        "iter-003 family brief seed 7: absorption conditioned on spread "
        "state; combines the two strongest iter-001/002 findings -- the "
        "flow_divergence_300s champion and the spread-state conditioning "
        "idea behind spread_z_300s."
    ),
    compute=compute,
)
