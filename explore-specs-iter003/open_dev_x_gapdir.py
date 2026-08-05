"""Explore-lane prototype spec (iter-003 round 3, family R3-A anchor deviation).

open_dev_x_gapdir: open-deviation (bps) signed by the overnight gap
direction -- does the intraday stretch CONTINUE the overnight move or
REVERSE it?
"""
import polars as pl

from hft_autofactor.explore import explore_prototype


def compute(part: pl.DataFrame) -> pl.Series:
    """dev_from_open_bps * sign(open - pre_close); defined from row 0."""
    dev = (pl.col("mid_px") - pl.col("open_px")) / pl.col("open_px") * 1e4
    gap_sign = (pl.col("open_px") - pl.col("pre_close_px")).sign()
    return part.select((dev * gap_sign).alias("value"))["value"]


PROTOTYPE = explore_prototype(
    name="open_dev_x_gapdir",
    mechanism=(
        "Two-timeframe agreement around the anchor pair. The SAME intraday "
        "deviation from the open means different things under different "
        "overnight backdrops: on a gap-UP day, a positive deviation "
        "EXTENDS the overnight repricing (previous-close-to-open trend "
        "still being honored -> continuation regime, drift persists); on "
        "a gap-DOWN day the identical positive deviation UNDOES the "
        "overnight move (gap-fill / reversal regime, where counter-"
        "pressure and profit-taking make persistence less likely). "
        "Multiplying by the gap sign folds the two cases onto one axis: "
        "positive = intraday flow agrees with overnight direction, "
        "negative = intraday flow fights it. The bare overnight gap is "
        "constant within the day (no intraday rank info) and the bare "
        "deviation ignores the overnight state; the product varies "
        "intraday and carries the agreement/disagreement question the "
        "round-2 overnight-gap x range-pos attempt (IS-dead) could not "
        "pose, because here the interaction is with the WINNING open-"
        "deviation axis, not the range-position axis."
    ),
    info_set="mid_px, open_px, pre_close_px",
    inspiration=(
        "iter-003 round-3 R3-A family brief direction 2 (interaction of "
        "pre-close deviation with the intraday open-deviation: gap-fill "
        "pressure vs trend continuation); retry of the dead R2 "
        "overnight_gap_x_range_pos question on the anchor axis that "
        "survived round 2."
    ),
    compute=compute,
)
