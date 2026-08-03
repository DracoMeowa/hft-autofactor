"""Pipeline subpackage: stage orchestration and the hftaf CLI."""
from .orchestrator import (
    JobResult,
    run_convert_stage,
    run_eval_stage,
    run_factor_stage,
    run_mask_stage,
)

__all__ = [
    "JobResult",
    "run_convert_stage",
    "run_eval_stage",
    "run_factor_stage",
    "run_mask_stage",
]
