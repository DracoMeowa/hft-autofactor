# 评估机制 v2：IC 为主 + 固定成本头部统计（2026-08-05，用户指令）

状态：**生效**。取代 #86 矩阵的 gate 地位（docs/design/eval-redesign-86.md）。
#86 条件矩阵代码保留，降级为**描述性**工件，不再作为任何准入依据。

## 规格（用户原话要点）

1. **准入 gate = IC**。因子在各 horizon 的 RankIC 显著性为准入主判据
   （沿用探索通道 screen：IS/OOS purged split，Newey-West t，retention，
   去重——这些不变）。
2. **交易成本固定单边 3bp**（round-trip 6bp）。fee_table_v1 的分场景
   费用栈、价差穿越+滑点执行模型、融券成本模型**全部不再参与 gate**。
   此前的 26-29bp/往返估计（含 taker 穿价差 + 每边 1tick 滑点）作废，
   按用户判断该套成本假设有误；本备忘不重新辩论。
3. **附带统计**：每个 horizon 输出头部做多/做空收益统计——
   因子值 top τ 分位做多、bottom τ 分位做空（τ ∈ 1%, 5%, 10%），
   报告 gross 与 net（net = gross − 6bp 往返）。描述性，不参与 gate。
4. **horizon 全开**：15s / 30s / 60s / 300s / 900s 一律评估，
   不因成本理由预先剔除任何 horizon（明确推翻 iter-001 后
   "避开 15s" 的记忆导向；该教训中 15s+taker 的结构性成本结论
   在本规格下不再适用，因成本假设已换）。
5. **资源纪律**：共享服务器 workers ≤ 4，内存受控（单标的面板很小，
   批量候选评估走 Python 探索通道，不做无谓 replay）。

## 实现落点

- `explore/screen.py`：ScreenConfig 增加 `cost_bps_per_side=3.0`、
  `head_taus=(0.01,0.05,0.10)`；新增 `head_long_short_stats()`
  （日内分位数切头尾，OOS 面板上计算）；screen 报告每个 horizon
  行附 `head_stats`。gate 判据仍为 IC（未变）。
- 台账照旧：screen 的每个 (factor, horizon) 先记账再看线。

## 与旧机制的关系

- #86 24 格矩阵、三佣金场景、融券成本：保留代码与历史工件
  （/data/factor_lzt/prod/matrix/），仅作描述性参考。
- iter-001 五因子的历史结论（taker 成本吞没）是**旧成本假设下**的
  结论，在 v2 下不作为否决依据；其 IC 层面的结论仍然有效。
