# iter-003 第二轮评估报告（588000，59 列宽表首批迭代）

日期：2026-08-06　批次：iter003_round2（#145）
前置：batch-2 宽表落地（#144，面板 42→59 列，66 交易日 × 314,053 行，verify_mat59.py 全 PASS）。

## 一句话结果

58 个候选 → eval-v2 门槛过 15 → 对 22 因子库做 pooled Spearman 去重砍 2 → **入库 13，库总计 35**。

## 批次构成（4 个研究组，各由一个 factor-researcher 子代理产出）

| 组 | 主题 | 个数 | 过筛 | 入库 |
|---|---|---|---|---|
| R2-A | 日内区间/OHLC 参照（open/high/low 新列） | 14 | 3 | 3 |
| R2-B | 短窗 OFI/TI 与流压力（ofi_15s/30s 等新列） | 16 | 5 | 4（1 个面板重复被门槛砍） |
| R2-C | 全簿 vs 五档背离 + 成交颗粒度（total_*_vol 新列） | 14 | 6 | 4（2 个被库内重复砍） |
| R2-D | IOPV 动态/累计/节奏（iopv_velocity 新列） | 14 | 2 | 2 |

流程耗时：注册 15 秒；run 58 个 × 66 天 ≈ 3.8 分钟（因果探针全过）；screen ≈ 4.2 分钟。

## 门槛（eval-v2）

RankIC 为主：每个 horizon IS+OOS Newey-West |t| ≥ 2.0；retention ≥ 0.5 且 IS/OOS 同号；对面板列 |Spearman| ≤ 0.85；批内/对库再做 pooled Spearman（阈值 0.85）。成本固定单边 3bp（往返 6bp），头部多空统计仅描述。划分：训练 60 天 / OOS 5 天 + 1 天禁运。

## 入库 13 个（按亮点排序）

| 因子 | 含义 | 通过 horizon | 最强 OOS IC（t） | 对面板 max|ρ| |
|---|---|---|---|---|
| **dev_from_open_bps** | 现价相对开盘价的偏离（bps） | **全 5 个** | 900s −0.4975（−8.79） | 0.239 |
| **top5_book_div_z_300s** | 五档与全簿深度背离的 z 值 | 15/30/60/900 | 15s **+0.2140（+14.20）** | 0.778 |
| range_pos_x_spread_z | 区间位置 × 价差状态交互 | 30/60/300/900 | 900s +0.2738（+3.84） | 0.193 |
| fullbook_imb_mom_60s | 全簿买卖量失衡 60s 动量 | 15/30 | 15s +0.0890（+6.46） | 0.603 |
| fullbook_imb_z_300s | 全簿失衡的 300s 窗口 z | 15 | 15s +0.0823（+10.43） | 0.581 |
| ofi_concord_15_60 | 15s 与 60s OFI 同向共振 | 15 | 15s +0.0863（+10.13） | 0.846 |
| ofi_15s_z_120s | 15s OFI 的 120s 窗口 z | 全 5 个 | 900s +0.0225（+5.33） | 0.790 |
| ti_accel_15_60 | TI 快减慢加速度 | 15/30/60 | 15s +0.0464（+3.91） | 0.651 |
| conc_imb_z_300s | 深度集中度失衡的 z | 300/900 | 900s +0.0840（+2.99） | 0.521 |
| ofi_per_depth_z_300s | 单位深度 OFI 的 z | 15 | 15s +0.0538（+4.43） | 0.774 |
| mid_roll_range_pos_300s | 滚动 300s 区间内位置 | 900 | 900s −0.1470（−3.20） | 0.540 |
| iopv_vel_drift_300s | IOPV 速度 300s 漂移 | 900 | 900s −0.1786（−2.90） | 0.371 |
| iopv_vel_z_300s | IOPV 速度的 z | 15 | 15s +0.0411（+2.67） | 0.847 |

头部多空（τ=5%，OOS 5 天，仅描述）：range_pos_x_spread_z 是唯一头部净收益为正且两边都正的（多头 +12.3bp / 空头 +18.0bp，900s）；其余多数头部毛收益扣 6bp 后为负——与第一轮结论一致，头部只作描述，不论证可交易性。dev_from_open_bps 900s 头部尾部极端（±60bp 量级）是强趋势日贡献，同样仅描述。

## 去重（对 22 个库内因子 + 批内，314,053 行 pooled Spearman）

砍 2 个（均为库内近克隆，规则：保先入库者）：

- **conc_imb_mom_60s**：ρ=0.898 对 depth5_delta_60s（第一轮入库）——深簿动量已在库。
- **top5_book_div_mom_60s**：ρ=0.874 对 oir_mom_60s（第一轮入库）——其 z 变体 top5_book_div_z_300s（与该克隆 ρ 仅 0.670、OOS IC 更强）存活，家族信号不丢。

批内新×新最高 ρ=0.779（恰好是两个被砍者之间），无需批内砍。
观察名单（0.65-0.80，未过线但记账）：mid_roll_range_pos_300s × rv_asym_300s 0.794；iopv_vel_drift_300s × rv_asym_300s 0.784；conc_imb_z_300s × top5_book_div_z_300s 0.778（水平 z vs 背离 z，问题不同，双留）；dev_from_open_bps × mid_day_range_pos 0.767（锚不同：开盘价 vs 高低区间）。

## 死因地图（43 个 FAIL）

按主导死因分组：

| 组 | IS 就死 | OOS 崩 | 面板重复 |
|---|---|---|---|
| R2-A 区间/OHLC | 3（隔夜跳空×区间位置、区间位置×OFI/WDI 交互） | 8（区间宽度/极值/斜率类：IS t 2-5 → OOS |t|<1.3） | — |
| R2-B 短窗流 | 2（事件强度 z 及其交互） | 8（ltns 大单净占比 3 个全灭、净买压 2 个、ofi_accel_z、ti_15s_z） | 2（ofi_accel_15_60 对 ofi_60s ρ0.863；ti_concord_15_60 对 trade_imbalance_60s ρ0.870） |
| R2-C 全簿/颗粒度 | 5（隐藏深度动量 2 个、成交笔数 z、成交均量动量、量节奏） | 3（成交均量 z 及其交互） | — |
| R2-D IOPV/累计 | 8（iopv_vel×ofi、各类累计/占比/节奏） | 4（事件强度 z、ofi 占比、强度 regime 交互、ti15 累计） | — |

关键死因结论：

1. **累计类是重灾区**（R2-D 12/14 死，多数 OOS 崩）：把 15/30s 流量简单累加 300s 学到的是样本内噪音；慢 horizon 要的是**状态**（区间位置、背离 z、漂移），不是原始累计。
2. **成交颗粒度全灭**（trade_size/trade_count/vol_tempo 0/6）：588000 上成交笔数/均量统计相对流不平衡没有增量信息。
3. **裸隐藏深度量不行，比值/z 才行**：hidden_imb（全簿减五档的裸量动量）IS 就死，而同族的归一化版本 fullbook_imb_z、conc_imb_z、top5_book_div_z 全过。
4. **快减慢加速度 ≈ 母列**：ofi_accel_15_60 数值上几乎就是 ofi_60s（ρ0.863 撞门槛）——做 z/accel/concord 变体前先想基底列是不是已经在库/面板里。
5. IOPV：**水平死（第一轮 0/8）、速度活**（本轮 2 个过）；套利压力的动态有信号，静态没有。

## 经验沉淀（给第三轮）

1. 日锚参照家族持续产出：dev_from_open_bps 全 5 horizon 通过且对面板相关仅 0.239——开盘价是继 mid_day_range_pos 之后第二个正交锚。还可以做：相对昨收偏离的动态化、开盘后 N 分钟斜率 × 当前偏离等。
2. 五档之外有真信息：top5_book_div_z_300s 是本轮 IC 最强（15s OOS 0.21），五档与全簿的背离值得继续挖（不同窗口的背离变化率、背离与流的方向一致性等）。
3. 新短窗列（ofi_15s/30s、ti_15s/30s）值得继续做 regime 化变体，但裸值/简单差分容易撞面板重复。
4. 已饱和区：累计类、成交颗粒度、隐藏深度裸量、事件强度 z——第三轮不再提。

## 归档位置

- library/candidates.json 键 `iter003_round2`（门槛/划分/13 明细含头部/2 砍掉/死因分组/经验）。
- scratch-iter003/：reports_round2/（15 份 PASS 报告）、round2_admitted_detail.json、round2_deathmap.json、iter003_round2_corr.json/.txt（服务器副本 /data/factor_lzt/iterations/）、本报告（服务器副本同处）。
- 服务器日志：/data/factor_lzt/logs/iter003_round2.log；评估脚本 /data/factor_lzt/scripts/iter003_round2.sh、round2_corr.py。
