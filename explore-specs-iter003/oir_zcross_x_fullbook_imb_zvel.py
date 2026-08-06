"""Explore-lane prototype spec (iter-003 R5, family R5-C).

oir_zcross_x_fullbook_imb_zvel: touch-queue crossing events WEIGHTED by
the full-book extreme velocity intensity. The oir crossing velocity
(event-sparse) multiplied by the fullbook imbalance extreme velocity
(continuous), testing whether touch hand-offs backed by broad-book
repositioning survive at longer horizons.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing z window
LAG = 5  # 5 x 3s rows = 15s velocity lookback


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def _fullbook_imb() -> pl.Expr:
    tb = pl.col("total_bid_vol").cast(pl.Float64)
    ta = pl.col("total_ask_vol").cast(pl.Float64)
    den = tb + ta
    return (
        pl.when(den > 0.0)
        .then((tb - ta) / den)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def compute(part: pl.DataFrame) -> pl.Series:
    """oir crossing velocity * fullbook extreme velocity; warm-up null."""
    zo = _z(pl.col("oir"), W)
    zo_lag = zo.shift(LAG)
    flip = (
        pl.when(zo.is_null() | zo_lag.is_null())
        .then(pl.lit(None, dtype=pl.Float64))
        .when((zo.sign() != zo_lag.sign()) & (zo != 0) & (zo_lag != 0))
        .then(1.0)
        .otherwise(0.0)
    )
    oir_cross_vel = flip * (zo - zo_lag)

    zf = _z(_fullbook_imb(), W)
    dzf = zf - zf.shift(LAG)
    fb_zvel = dzf * zf.abs()

    return part.select((oir_cross_vel * fb_zvel).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="oir_zcross_x_fullbook_imb_zvel",
    mechanism=(
        "Broad-book-confirmed touch crossings: oir_z_cross_vel_15s scores "
        "the cheapest-quotes regime hand-off -- the fastest, cheapest place "
        "to signal urgency, but also the easiest to fake (a single small "
        "order can flip the touch). The crossing velocity alone does not "
        "distinguish an informed queue takeover from a marginal touch "
        "rotation. Hypothesis: weighting each touch crossing by the "
        "CONCURRENT full-book extreme velocity (dz_fullbook * |z_fullbook|) "
        "ranks touch hand-offs by whether the entire patient book is "
        "simultaneously relocating. A touch crossing where the FULL book "
        "(all levels, institutional inventory) is moving at extreme speed "
        "in the new direction is a genuine whole-market regime change "
        "visible from touch to tail -- the highest-conviction signal, "
        "surviving at 300-900s. A touch crossing where the full book is "
        "quiescent is just the tip reshuffling with no broad backing, "
        "likely noise. This spans the WIDEST scale pair in the cross-base "
        "cluster: oir = 1 level (touch), fullbook = all levels -- the "
        "agreement of the fastest surface with the broadest depth. "
        "Distinct from fullbook_imb_zcross_x_wdi_zvel (broad-book crossing "
        "× top-5 velocity): different event base and different continuous "
        "backer, asking which direction of confirmation (touch backed by "
        "broad, or broad backed by top-5) carries more long-horizon signal."
    ),
    info_set="oir, total_bid_vol, total_ask_vol",
    inspiration=(
        "iter-003 R5-C family brief direction 5 (cross of TWO z-vel "
        "bases): touch crossing × fullbook extreme velocity. Spans the "
        "widest book-scale pair (1-level vs all-level) for a synchronization "
        "test across the full market depth."
    ),
    compute=compute,
)
