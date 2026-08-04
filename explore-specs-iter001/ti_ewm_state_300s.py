"""Explore-lane prototype spec (iter-001, flow-queue lens).

ti_ewm_state_300s: slow accumulation state of aggressive executed flow.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

# half-life 150s = 50 rows of the 3s grid -> per-row EWMA decay
ALPHA = 1.0 - 0.5 ** (1.0 / 50.0)
MIN_SAMPLES = 100  # 300s warm-up before the state is trusted


def compute(part: pl.DataFrame) -> pl.Series:
    x = pl.col("trade_imbalance_60s")
    return part.select(
        x.ewm_mean(alpha=ALPHA, adjust=False, min_samples=MIN_SAMPLES,
                   ignore_nulls=True).alias("value")
    )["value"]


PROTOTYPE = explore_prototype(
    name="ti_ewm_state_300s",
    mechanism=(
        "Flow-accumulation state variable: an exponentially-weighted running "
        "integral of signed aggressive trade imbalance with ~150s half-life. "
        "Meta-orders (institutional execution, arbitrage baskets vs IOPV) "
        "arrive over minutes, so the integrated imbalance of executed "
        "aggressor flow is a persistent pressure/information state, not a "
        "60s impulse. High positive state = sustained net buying aggression "
        "accumulated over the last several minutes -> continuation expected "
        "at 60-300s horizons, where the library's fixed 60s windows have "
        "already decayed to their tails."
    ),
    info_set="trade_imbalance_60s (library)",
    inspiration=(
        "Digest iter-000: '300s HORIZON = WEAKEST AND MOST OPEN ... every "
        "library flow factor uses a 60s lookback, so 300s only sees decay "
        "tails. Proposals with 2-5 min state variables (flow accumulation) "
        "directly target this gap'; meta-order propagation and square-root "
        "impact (Bouchaud-Gefen-Potters-Wyart 2004, G(t) ~ t^-1/2)."
    ),
    compute=compute,
)
