# #86 评估准则改造设计规格 — 双轨评估 + 多情境条件盈利矩阵

> 日期：2026-08-05 · 状态：**提案（待评审）**
> 触发：iter-001 教训④（单侧门槛结构性拒绝负 IC 反转信号）+ iter-001 回测报告
> （`/data/factor_lzt/prod/reports/backtest_588000_iter001.md`：78/78 run 择时 alpha 为负、
> 利润全为 carry beta；taker 往返成本 28–41bp 对 15s 信号是死刑）+ 用户指令
> （多空分解、top/bottom 阈值入场、波动率 regime 条件化、IC 与盈利能力双轨并行）。
> 本规格是"宪法"：之后所有批量试验的评测按此记账与裁决。

---

## 1. 目标与非目标

**目标**
1. A 轨（IC 准入）改**双侧**：负 IC 不再被拒，方向 = sign(IC)（反转信号合法化）。
2. B 轨（盈利能力）新增**条件矩阵**：方向 × 信号强度阈值 τ × 波动率 regime 的
   逐笔净边缘统计，回答"做多 top τ 信号 / 做空 bottom τ 信号在什么情境下能赚钱"。
3. **净成本条件边缘 gate**：准入因子的经济学裁决从"底仓结构净 Sharpe"改为
   "主情境格 carry-free 净边缘 > 0"。
4. 空头腿成本入费用栈（融券费率，#129 调研）。

**非目标（本期不做，留接口）**
- maker/挂单执行模型（iter-001 回测报告建议②；v2 单独立项，把往返成本压到 ~5bp）。
- 引擎级空头模拟（融券借券库存链、T+1 还券约束）——v2；本期空头腿只做
  trade-level 统计评估。
- 横截面（多资产）评估——多资产扩容阶段。

## 2. 双轨架构

```
因子候选 ──► A轨 IC 准入（便宜、稳健）──► B轨 条件矩阵（经济学裁决）──► 全模拟验证（存活者）
             改双侧 |t|≥hurdle              trade-level 净边缘              现有引擎 + carry 剥离
             direction=sign(IC)             24 格预注册                      净择时 alpha > 0
             合成价值保留                    honest-N / BHY                   （v2: 空头模拟+maker）
```

- **A 轨不替代**：整体 IC 是合成价值维度（iter-001 报告 §6 教训④：三个 900s 反转
  信号 IC 显著为负，对合成有意义）。A 轨裁决"信号存在且方向可信"，B 轨裁决
  "按指定执行方式能否赚钱"。两轨独立记账。
- **顺序**：先 A 轨（全 66 天 IS/OOS 照预注册分割），通过者进 B 轨；B 轨主格通过
  才进全模拟。探索原型（未晋升 C++）可直接进 B 轨——因子列来自 explore panel
  join 或 derived registry 在线重算，不需要引擎物化。

## 3. A 轨改动（eval/gating.py）

1. `stage1_screen`：`t_stat_nw >= hurdle` → `abs(t_stat_nw) >= hurdle`；输出加
   `direction`（= sign(mean_ic)，±1）。其余不动（BHY p 值本就双侧、噪声地板比
   |IC|、|ICIR| 已双侧）。
2. `stage2_oos_gate`：`oos t >= min_oos_t` → `abs(oos t) >= min_oos_t`；sign_ok
   已有（IS/OOS 同号），win_rate 按方向校正后沿用。
3. 兼容性：ledger 记录新增 `direction` 字段；旧条目不受影响。阈值本身
   （HL hurdle、FDR q、retention、噪声地板）**一律不调**——改方向语义不是放松门槛。

## 4. B 轨：条件矩阵定义

### 4.1 维度（冻结，24 格 / 每 (因子, horizon)）

| 维度 | 取值 | 说明 |
|---|---|---|
| direction | {long, short} | long = 做多 top 信号；short = 做空 bottom 信号 |
| τ（入场阈值） | {0.005, 0.01, 0.05, 0.10} | 仅当 \|z\| ≥ 当日 trailing (1−τ) 分位才入场 |
| regime | {all, vol_q80+, vol_q90+} | 按日波动率的 trailing 分位条件化（§4.4） |

**主格（primary cell，预注册）**：direction = A 轨 direction，τ = 0.01，regime = all。
**只有主格参与准入裁决**；其余 23 格是描述性证据，防止"矩阵里挑最好格子"的
后验选择过拟合。新增格子必须修订本规格并全量重记账。

### 4.2 信号与入场（因果纪律）

- z = `causal_zscore(factor, 100 行)`（复用现有实现，trailing，跨日不携带）。
- τ 阈值：当日**日内 expanding trailing 分位**——第 i 行的阈值 q_{1−τ} 只用
  该行之前（含）的 |z| 样本计算；warm-up = 当日前 1600 行（≈80 分钟）不入场。
  选择日内而非跨日分位：不引入跨日状态、自动适应当日波动 regime。
  （单测保证：修改未来行不改变任何行的入场判定。）
- 决策滞后：与引擎一致，signal_lag_rows = 1（入场判定用 i−1 行的 z）。

### 4.3 Trade 定义（非重叠、carry-free）

- 候选行：入场条件满足 ∧ tradable（复用引擎语义：flags bit0/1 干净、双边报价、
  距 session 边界 ≥ H）∧ 标签 fwd_mid_ret_H 非空。
- **非重叠贪心**：按 ts 扫描，入选 trade 需 ts ≥ 上一笔 ts + H。每笔 trade =
  t 入场、首个 ≥ t+H 的快照结算——与标签同解析点，天然无重叠、无持仓链、
  无结构性 carry。这是与 iter-001 底仓回测的本质区别。
- 每格样本：OOS 段（预注册 20250901–20250930）为主裁决区；IS 段产出仅作
  对照描述。

### 4.4 Regime 条件化（波动率）

- 日波动率：rv_day(d) = 当日 3s mid 收益序列的 std（从 panel 计算）。
- 分位：**trailing 20 个交易日**窗口内 rv_day 的分位（只用 < d 的日子）。
  vol_q80+ = 前 20% 高波动日；vol_q90+ = 前 10%。前 20 个交易日 warm-up 期
  regime 只能取 all。
- **样本量门禁**：pilot 66 天里 q90+ 格 ≈ 5–7 天。regime 格 n_days < 10 或
  n_trades < 100 ⇒ `descriptive_only=true`，只出数字不做 gate。当前数据量下
  regime 结论一律描述性——机制先建好，数据量到位后自动解锁。

### 4.5 每格 metrics

| 指标 | 定义 |
|---|---|
| n_trades / n_days | 样本量 |
| mean_net_edge_bps | 单笔净边缘均值（bp，对入场名义） |
| t_nw_daily | 按日聚合 trade 边缘 → 日均边缘序列 → Newey-West t（max_lag=5） |
| hit_rate | 净边缘 > 0 的 trade 占比 |
| gross_edge_bps | 未扣成本边缘（对照） |
| cost_bps | 实际扣掉的单笔成本均值（§4.6） |
| valid | n_trades ≥ 100 ∧ n_days ≥ 10 ∧ regime 达标 |

### 4.6 成本模型（每笔、逐行实价）

- **taker 基线**（v1 唯一执行假设，与 iter-001 回测一致口径）：
  单边成本 = |fill − mid|（fill = 对穿报价 ± 1 tick，与引擎 `cross_spread_fill`
  同式）+ `side_cost_cny` 费用栈 + 深度冲击 overlay（引擎同参数）。
  long 买入按 ask 穿价；short 卖出按 bid 穿价。
- **空头融券成本**（#129 参数）：`ShortCostModel(borrow_rate_annual,
  min_charge_days)`。单笔空头成本 = notional × rate × max(H_seconds,
  min_charge_days × 86400) / 年化秒数。**若调研证实最低按 1 天计息，15s
  horizon 的空头腿要多背 ~rate/250 的成本——这可能直接否决短 horizon 空头，
  是调研的关键产出。** 参数未入库前矩阵短腿标 `cost_model=provisional`。
- 三佣金情景（institutional / retail_negotiated / retail_default）全跑；
  主格 gate 要求三情景**全部**存活（沿用现有惯例）。

### 4.7 主格 gate（净成本条件边缘）

全部满足才算 B 轨通过：
1. `mean_net_edge_bps > 0`（三情景）；
2. `t_nw_daily ≥ 2.0`（OOS 段）；
3. `valid = true`；
4. `gross_edge_bps > cost_bps`（边缘必须大于成本本身，即"成本覆盖率 > 1"）。

## 5. 全模拟验证（B 轨通过者的第二级）

沿用现有引擎 + iter-001 报告 §8 的归因方法（per-day avg_position_units 已在
产物中）：
1. long 腿全模拟（底仓结构 dzit2 与 lf 两式）；
2. **新增 gate：carry 剥离后净择时 alpha > 0**——直接回应 iter-001 "利润全是
   carry beta" 的发现；carry 口径照抄报告附录（Σ 日均持仓×(收−开) + 隔夜缺口）。
3. 空头腿全模拟与 maker 执行：v2，接口预留（`ExecutionModel` 抽象位）。

## 6. 防过拟合与记账纪律

1. 24 格 + 主格身份**运行前**写入 matrix config（JSON），随报告归档。
2. 每格评估在 OOS 上读取一次之前，先以 stage="matrix_cell" 记入 TrialLedger
   （honest N 继续全量计数；HL hurdle 随 N 自动抬升）。
3. 跨格 BHY-FDR（q=0.10，双侧 p）作为报告列，不作为主格 gate 的豁免理由。
4. 禁止事后换主格、事后增删格、事后调 τ 集合。违规 = 该因子整轮作废。

## 7. 落点与工件

| 工件 | 位置 |
|---|---|
| 矩阵构建器（纯函数） | `python/hft_autofactor/backtest/conditional.py` |
| CLI | `hftaf-backtest --matrix`（新 flag：`--taus`、`--regimes`、`--direction auto\|long\|short`、`--matrix-out`） |
| 因子列来源 | parquet 库因子 / derived registry（depth5_delta 等）/ explore panel join（反转 trio） |
| 融券参数 | `docs/knowledge/etf_backtest_params.yaml` 新增 `securities_lending` 节（+镜像同步） |
| 报告 | `{out}/matrix/{factor}_h{H}/matrix.json + matrix.md`（24 格全表 + 主格裁决 + config 快照） |
| 测试 | `test_bt_conditional.py`（合成信号多空腿、trailing 分位因果性、非重叠、regime 旗标、融券成本）；`test_gating.py` 双侧化用例 |

## 8. 验收标准

1. 合成数据：带已知正边缘的因子 → 主格（long, τ=0.01, all）净边缘 > 0 且
   short 腿 ≈ 0/负；纯噪声因子 → 所有格 |边缘| 低于其置换地板；反转因子
   （负 IC）→ A 轨 direction=−1 通过、short 腿主格净边缘 > 0。
2. 未来行扰动测试：篡改因子序列未来段，任何行的入场判定与边缘不变。
3. iter-001 五因子复测（#130）产出完整矩阵报告；depth5_delta/flow_divergence
   的预期：**taker 成本下 15s 主格大概率不存活**（往返 28–41bp），这是规格
   正确性的体现而非失败——存活与否如实归档，供 maker 模型（v2）复测。
4. 全套 pytest 绿；ledger 新增条目格式向后兼容。

## 9. 与后续工作的接口

- **iter-002 批量**：每批候选过 A 轨后自动进 B 轨矩阵；批次报告含 24 格全表。
- **物化列**（wishlist）：large_trade_share / trade_arrival_burst 解锁后同流程。
- **maker 执行（v2）**：`conditional.py` 的成本模型参数化（`ExecutionModel`
  注入），maker 落地后 15s 因子复测。
- **多资产扩容**：矩阵加 cross-sectional τ（横截面分位）维度，规格追加。

---

## 修订记录

### 修订 1（2026-08-05，#129 融券调研落地后）

1. **§4.6 空头成本模型定稿**（参数已入 `etf_backtest_params.yaml` 的
   `securities_lending` 节，权威件在 `docs/knowledge/`）：
   - 计息口径：**自然日 / 360**，最低计费 **1 个自然日**（当日借也按整天）。
     原文"年化秒数"的草图作废；实现为 `costs.short_borrow_cost_bps`。
   - **T+1 还券（融资融券交易实施细则 art.16）**：融券卖出后次一交易日才可
     买券还券 ⇒ 空头最短持有 = 隔夜。空头格的 H 秒标签**不含隔夜缺口风险**；
     该风险在 config 快照 `short_settlement: T+1_repay` 中记载，不定价。
   - **涨幅跌停报价规则（uptick）**：融券卖出申报价 ≥ 最新成交价 ⇒ 矩阵空头
     入场 fill 取 `max(穿价 bid 成交, ceil(last_px))`。
   - 默认费率 **8%/年**（保守；当前市场 4–6%，历史挂牌 10.6%，敏感性带
     4–10.6%）。588000 自 2020-11 上市即两融标的；无分红历史（无补偿成本）；
     转融券 2024-07 起暂停（券源稀缺）；保证金比例 ≥80%（2023-10-30 起）。
   - 预期后果：horizon ≤ 900s 的空头腿每笔多背 ≥ ~2.2bp（8%/360）借券费，
     15s 空头大概率不经济——这是经济现实的如实反映，不是矩阵故障。
2. **CLI 定形**：`hftaf-backtest --matrix --direction ±1 [--matrix-out DIR]
   [--eval-dates SPEC]`。τ 集合与 regime 集合在代码中冻结（`MatrixConfig`
   默认值，即 §4.1），**不设 CLI 覆盖入口**——§6.4 禁止事后改动，需要覆盖
   时必须修订本规格并全量重记账。`--direction` 即 A 轨 ic_direction；
   从 stage1 工件自动推导（auto）留给 iter-002 流水线接线。
3. 空头腿在矩阵报告中的 cost_model 标签：有融券参数 = `taker_v1+borrow`；
   参数缺失 = `taker_v1_no_borrow_PARAMS_PENDING` 且强制 descriptive_only。
