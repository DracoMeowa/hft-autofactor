"""Evaluation subpackage: IC metrics, day-blocked splits, statistical gating."""
from .gating import (
    GateConfig,
    TrialLedger,
    bhy_critical_values,
    deflated_sharpe_pvalue,
    permutation_noise_floor,
    stage1_screen,
    stage2_oos_gate,
    t_hurdle,
)
from .ic import (
    ICStats,
    ic_stats,
    newey_west_n_eff,
    rank_ic_cross_section,
    rank_ic_time_series,
    spearman,
)
from .splits import Split, is_oos_retention, purged_day_splits

__all__ = [
    "GateConfig",
    "ICStats",
    "Split",
    "TrialLedger",
    "bhy_critical_values",
    "deflated_sharpe_pvalue",
    "ic_stats",
    "is_oos_retention",
    "newey_west_n_eff",
    "permutation_noise_floor",
    "purged_day_splits",
    "rank_ic_cross_section",
    "rank_ic_time_series",
    "spearman",
    "stage1_screen",
    "stage2_oos_gate",
    "t_hurdle",
]
