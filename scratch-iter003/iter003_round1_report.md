# iter-003 第一轮：60 个宽表因子候选 评估 + 去重 + 入库

日期：2026-08-05　标的：588000　面板：66 个交易日（20250701–20250930），314,053 行

## 一句话结论

60 个新因子里 19 个过筛、41 个死掉；过筛的再做批内去重，砍掉 3 个近重复，**最终入库 16 个**（加上之前 6 个，因子库共 22 个）。最强的是"盘口失衡动量"家族（OIR/WDI/深度 30–120 秒动量），15 秒样本外 IC 0.12–0.17；mid_day_range_pos（日内区间位置）是唯一 5 个 horizon 全过、且和库里所有因子都不相关的**全新信号源**。

## 门槛设置（eval-v2）

- 准入：IS 和 OOS 的 RankIC Newey-West |t| 都 ≥ 2.0；retention（OOS/IS 之比）≥ 0.5 且 IS/OOS 符号必须一致（翻符号直接死）；与面板 17 个因子列的相关系数 ≤ 0.85。
- 划分：训练 60 天（20250701–20250922），测试 5 天（20250924–20250930），中间隔 1 天。
- 成本：固定单边 3bp（往返 6bp）。头部多空收益统计只作描述，不作门槛。
- Horizon：15/30/60/300/900 秒全开。

## 运行流水

注册 81 个原型 → 跑数 60/60 成功（约 2.5 分钟）→ 评估 19 过 / 41 死（约 3 分钟）→ 批内 Spearman 去重（25 列、300 对、31 万行池化）→ 砍 3 留 16。

## 入库的 16 个（OOS IC / t 值，只列过门槛的 horizon）

| 因子 | 与面板最大相关 | 通过的 horizon（OOS IC / t） |
|---|---|---|
| oir_mom_60s（失衡比 60s 动量） | 0.656 | 15s: +0.172/t14.3　30s: +0.129/t10.2 |
| top_book_delta_120s（顶档差 120s 变化） | 0.674 | 15s: +0.171/t15.4　30s: +0.128/t12.5 |
| wdi_mom_30s（加权深度失衡 30s 动量） | 0.558 | 15s: +0.167/t9.9　30s: +0.125/t7.4 |
| wdi_mom_180s | 0.572 | 15s: +0.157/t16.0　30s: +0.124/t18.7 |
| depth5_delta_30s（五档深度差 30s 变化） | 0.357 | 15s: +0.109/t7.5　900s: +0.032/t6.0 |
| wdi_accel_90s（失衡加速度） | 0.315 | 15s: +0.089/t5.8　30s: +0.068/t4.4　60s: +0.065/t5.2　900s: +0.028/t5.9 |
| last_mid_gap_ticks（成交价偏离中间价） | 0.307 | 15s: +0.054/t5.5 |
| mid_day_range_pos（日内区间位置） | 0.407 | **全过**：15s -0.043 → 900s -0.343（越靠近日高，未来越跌） |
| ofi_z_x_spread_z（订单流 × 宽价差状态） | 0.432 | 15s: -0.047/t-3.9 |
| flow_divergence_120s | 0.345 | 15s/30s/60s: +0.033~0.037 |
| flow_divergence_60s | 0.245 | 15s: +0.033　60s: +0.039/t2.9 |
| flow_divergence_x_spread_z | 0.143 | 15s: -0.027/t-2.2 |
| depth5_delta_120s | 0.402 | 300s: +0.049/t2.5　900s: +0.060/t2.9 |
| log_mid_ret_120s | 0.616 | 900s: -0.091/t-2.4 |
| price_accel_60_180（价格加速度） | 0.206 | 900s: +0.086/t2.4 |
| signed_rv_60s（带符号已实现波动） | 0.782 | 900s: -0.059/t-2.3 |

头部描述（OOS 5 天，τ=5% 分位）：最强的 oir_mom_60s / top_book_delta_120s / wdi_mom_30s/180s 多头端毛收益 +2bp 上下、空头端也 +2bp 上下，扣掉 6bp 往返成本后为负——**头部统计只作描述**，因子的价值在排序和条件化，不在靠头部裸收益覆盖成本（这与上次 rescreen 结论一致）。

## 批内去重砍掉的 3 个

| 砍掉 | 原因 | 保留 |
|---|---|---|
| microprice_dev_mom_60s | 与 oir_mom_60s 相关 0.996，IC 一模一样 | oir_mom_60s（机制更直接） |
| vol_adj_mom_60s | 与 signed_rv_60s 相关 0.974，且与库里 ofi_60s 相关 0.849（换皮） | signed_rv_60s |
| top_book_delta_30s | 与 wdi_mom_30s 相关 0.878 | wdi_mom_30s（面板相关更低、OOS IC 更高；顶档家族留 120s 版） |

另有 microprice_dev_z_300s 被评估阶段的去重门槛直接拒掉（与库里 microprice_dev 相关 0.902，虽然它 15/30s 的 IC 本身很好）——说明"机制上像"的变体必须查相关再提交。

## 41 个失败因子的死因分类

| 死因 | 数量 | 代表 |
|---|---|---|
| premium/IOPV 家族制度断裂（IS t -3~-8 → OOS |t|≤1.5） | 8 | iopv_premium_z_120s/600s、premium_dev_day、ofi_x_premium_sign |
| 短窗裸动量 retention 不过/翻符号 | 5 | log_mid_ret_15s/30s、gap_x_direction |
| 深度/斜率水平类 IS 就死或翻符号 | 5 | book_slope_z_300s、depth_ratio_5to1_z、queue_pressure_x_slope（IS t7.2 但 retention 0.35） |
| OFI 窗口/动量变体 OOS 崩 | 5 | ofi_fast_slow、ofi_mom_60s（IS t5-10 → OOS |t|≤1.7） |
| 主动成交累积/交互 retention 崩（0.10-0.20） | 5 | ti_accum_120s、ti_z_x_spread_z |
| 已实现波动变体 IS 就死 | 5 | rv_z_300s、rv_ratio_60_300 |
| 价差水平本身没预测力 | 2 | spread_z_60s/120s（但价差作**条件**的交互项过了） |
| 大单份额类 IS 死 | 2 | large_share_mom_300s |
| 时段/制度交互 OOS 崩 | 2 | session_u_x_mom |
| 与库内重复（0.902） | 1 | microprice_dev_z_300s |
| 翻符号 | 1 | size_x_direction |

## 经验（第二轮必须遵守）

1. **盘口失衡动量/变化是 15/30s 最强家族**（16 个里占 8 个），OOS IC 0.09–0.17、t 5–15。但这一家族已经比较饱和，第二轮只补"增量"变体，先查批内相关再交。
2. **同家族冗余极大**：19 个过筛的里 3 个是批内重复（相关 0.88–0.996）。同一构造换个底（oir 动量 vs microprice 动量）可以几乎一模一样。
3. **mid_day_range_pos 是新大陆**：唯一 5 个 horizon 全过、与一切现有因子相关 ≤0.41。负 IC（靠近日高 → 未来收益低，均值回归）。第二轮要展开：区间扩张速度、靠近日内低点、当日已走幅度的变体。
4. **premium/IOPV 无条件因子别再做了**：0/8 全灭，两次确认制度断裂。只留"制度条件化"的组合尝试。
5. **SSE 上 cancel_ratio_60s 和 order_arrival_60s 全是 NaN**：凡是乘它们的因子（cancel_x_ofi、prem_x_cancel、signed_arrival_z）结构性死，SSE spec 里永远不许再用。
6. **短窗裸动量不行**：log_mid_ret_15s/30s retention 不过；短 horizon 的动量必须挂盘口/流信息。300/900s 的赢家都是慢状态变量。
7. **价差本身无方向预测力，但能当条件**：ofi_z_x_spread_z、flow_divergence_x_spread_z 都过了 15s。方向：用状态给流做条件。
8. **头部收益大多 1–6bp 毛收益、扣 6bp 成本为负**：头部统计保持描述性质，别拿它当门槛或拿它论证可交易性。

## 归档位置

- 候选库：`library/candidates.json` 新键 `iter003_round1`（门槛、划分、16 个入库明细含头部统计、3 个去重砍掉、死因分组、经验）。
- 原始评估报告：`scratch-iter003/reports_round1/`（60 个 JSON）；批内相关矩阵 `iter003_round1_corr.json/.txt`（服务器副本在 `/data/factor_lzt/iterations/`）。
- 本报告：`scratch-iter003/iter003_round1_report.md`（服务器副本在 `/data/factor_lzt/iterations/`）。

## 第二轮方向（预告）

- 展开 mid_day_range_pos 家族（区间扩张速度、距日内低点、已走幅度占比）。
- 状态条件化流：价差状态 × OFI/flow divergence 的不同窗口、深度状态条件。
- 300/900s 慢变量：深度变化的更长窗积累、跨日开盘漂移、隔夜缺口相关。
- 等宽表扩容（#140：总委买/委卖量、逐笔 15/30s 窗口、主动买卖量等 ~17 列）落地后，再开需要新列的候选。
