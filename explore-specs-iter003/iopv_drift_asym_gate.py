"""Explore-lane prototype spec (iter-003 R4-A, spread-z gating fill-in).

iopv_drift_asym_gate: sustained IOPV drift x a SIGN-ASYMMETRIC spread gate
-- wide stress weighted double the tight-comfort weight, NO sign flip in
either regime. Pure arbitrage-cost modulation of a same-signed drift.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 100  # 100 x 3s rows = 300s trailing window (base and gate)


def _z(x: pl.Expr, w: int) -> pl.Expr:
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """drift x (wide_depth + 0.5 x tight_depth); warm-up null.

    wide_depth = clip(sp_z, 0, inf); tight_depth = wide_depth - sp_z =
    clip(-sp_z, 0, inf). Both regimes keep the drift's own sign.
    """
    base = pl.col("iopv_velocity").rolling_mean(window_size=W, min_samples=W)
    sp_z = _z(pl.col("quoted_spread_ticks").cast(pl.Float64), W)
    wide = sp_z.clip(lower_bound=0.0)
    tight = wide - sp_z
    gate = wide + 0.5 * tight
    return part.select((base * gate).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="iopv_drift_asym_gate",
    mechanism=(
        "Arbitrage-cost MODULATION without regime sign flip: the sustained "
        "300s NAV drift keeps its own sign in BOTH quoting regimes but is "
        "weighted by how far the spread state sits from normal -- full "
        "weight per unit of wide stress, half weight per unit of tight "
        "comfort. The economic claim is pure execution-cost latency: "
        "arb that closes the ETF-vs-anchor gap is more expensive when "
        "spreads are wide, so drift continuation at 300-900s is STRONGER "
        "under stress; when quotes are unusually tight the same drift is "
        "partially arbitraged away on the fly, so continuation is "
        "attenuated but NOT reversed -- there is no price at which a "
        "sustained fundamental trend becomes a contrarian signal merely "
        "because quoting is comfortable. This is the sign-asymmetric "
        "counterpart of iopv_drift_x_spread_z: the product additionally "
        "asserts a tight-regime fade, this spec asserts only monotone "
        "cost modulation. Dedup note: algebraically gate = 0.75|sp_z| + "
        "0.25 sp_z, between the product (sp_z) and a symmetric |sp_z| "
        "gate -- sibling corr with the product may run high when tight-"
        "regime variance is small; the no-flip assertion is the distinct "
        "economic input."
    ),
    info_set="iopv_velocity, quoted_spread_ticks",
    inspiration=(
        "iter-003 R4-A fill-in brief: sign-asymmetric-gate variant; "
        "arbitrage execution cost is monotone in spread width (Stoll "
        "2003), so drift continuation should be regime-monotone, not "
        "sign-flipping; complements the product and one-sided gates of "
        "the same base."
    ),
    compute=compute,
)
