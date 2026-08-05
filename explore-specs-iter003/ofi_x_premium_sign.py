"""Explore-lane prototype spec (iter-003, flow-interaction lens).

ofi_x_premium_sign: book flow z-score gated by the SIGN of the ETF's IOPV
premium -- book building aligned with (or against) ETF mispricing.
"""
import polars as pl

from hft_autofactor.explore import explore_prototype

W = 60  # 60 x 3s rows = 180s ofi z window


def _z(x: pl.Expr, w: int) -> pl.Expr:
    """Trailing causal z-score; constant windows map to 0.0 (neutral)."""
    m = x.rolling_mean(window_size=w, min_samples=w)
    s = x.rolling_std(window_size=w, min_samples=w)
    z = (x - m) / s
    return pl.when(s.is_not_null() & (s == 0.0)).then(pl.lit(0.0)).otherwise(z)


def compute(part: pl.DataFrame) -> pl.Series:
    """z(ofi_60s, 180s) x sign(iopv_premium)."""
    ofi_z = _z(pl.col("ofi_60s"), W)
    prem_sign = pl.col("iopv_premium").sign()
    return part.select((ofi_z * prem_sign).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="ofi_x_premium_sign",
    mechanism=(
        "Arb-aligned book building: iopv_premium's SIGN says whether the "
        "ETF trades rich (premium -> creation arbitrage pulls it down/"
        "underlying basket demand) or cheap (discount -> redemption "
        "arb). Book flow ALIGNED with the arb direction (bid-side "
        "building at a discount, ask-side at a premium) flags AP/"
        "authorized-participant creation-redemption activity -- informed, "
        "mechanistic flow that resolves the mispricing and predicts the "
        "mean-reversion path at 30-300s. Book flow AGAINST the arb sign "
        "is noise fighting the primary ETF mechanic. This is the signed-"
        "gate repair of prem_x_ofi: the dead version multiplied two "
        "LEVEL z-scores; the sign gate keeps only mispricing DIRECTION "
        "and lets flow intensity vary freely."
    ),
    info_set="ofi_60s, iopv_premium (library)",
    inspiration=(
        "iter-003 family brief seed 15; iter-001 post-mortem: prem_x_ofi "
        "(level x level) IC ~ 0 -- retry with sign conditioning per the "
        "flow-works-when-SIGNED meta-lesson; ETF creation/redemption "
        "arbitrage mechanics."
    ),
    compute=compute,
)
