"""hft_autofactor - evaluation/validation layer for the HFT ETF factor pipeline.

Stages owned by this package:
  Stage 2 - lookahead mask validation (truncate-and-recompute prefix identity)
  Stage 3 - CSV -> parquet conversion under /data/factor_lzt
  Stage 4 - IC/RankIC evaluation, multiple-testing gating, IS/OOS splits
"""

__version__ = "0.1.0"
