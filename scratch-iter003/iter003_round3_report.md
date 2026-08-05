# iter-003 第三轮评估报告（588000，59 列宽表持续迭代）

日期：2026-08-06　批次：iter003_round3（#146）
前置：第二轮入库 13（库 35）；本轮按 round-2 经验定向挖四个方向。

## 一句话结果

57 个候选 → eval-v2 门槛过 19 → 对 35 因子库做 pooled Spearman 去重砍 9 → **入库 10，库总计 45**。

## 批次构成（4 个研究组，各由一个 factor-researcher 子代理产出）

| 组 | 主题 | 个数 | 过筛 | 入库 |
|---|---|---|---|---|
| R3-A | 日锚偏离的动态化（开盘价参照的衍生） | 14 | 4 | **0（全被去重砍：锚家族饱和）** |
| R3-B | 深簿背离分解（五档之外深度的单边拆解） | 14 | 7 | 6 |
| R3-C | 跨尺度流结构（快慢流的方向一致性/持续性） | 15 | 4 | 2 |
| R3-D | 存量信号的状态交互（已入库因子 × 状态门控） | 14 | 4 | 2 |

流程耗时：注册 15 秒；run 57 个 × 66 天 ≈ 4 分钟（因果探针全过）；screen ≈ 4 分钟。

## 门槛（eval-v2，同前两轮）

RankIC 为主：每个 horizon IS+OOS Newey-West |t| ≥ 2.0；retention ≥ 0.5 且 IS/OOS 同号；对面板列 |Spearman| ≤ 0.85；批内/对库再做 pooled Spearman（阈值 0.85，314,053 行 × 66 天）。成本固定单边 3bp（往返 6bp），头部多空统计仅描述。划分：训练 60 天 / OOS 5 天 + 1 天禁运。

## 入库 10 个（按亮点排序）

| 因子 | 含义 | 通过 horizon | 最强 OOS IC（t） | 对面板 max\|ρ\| |
|---|---|---|---|---|
| **div_z_x_spread_z** | 深簿背离 z × 价差压力 z | **全 5 个** | 15s −0.1701（−12.72） | 0.420 |
| **dev_open_x_spread_z** | 开盘偏离 × 价差压力 z | **全 5 个** | 900s +0.3045（+7.24） | 0.216 |
| conc_imb_x_spread_z | 深度集中度失衡 × 价差 z | 15/30/60/300 | 60s −0.0539（−11.01） | 0.265 |
| ofi_z_cross_vel_15s | OFI 的 z 与 15s 速度的背离 | 15/30/300/900 | 15s +0.0368（+3.93） | 0.327 |
| div_x_vis_share | 深簿背离 × 可见深度占比 | 30/60/300 | 60s +0.0319（+5.18） | **0.074** |
| div_pos_frac_300s | 背离的带符号占比形式 | 300/900 | 300s +0.1004（+2.70） | 0.471 |
| hidden_ask_supply_z_300s | 卖方隐藏深度供给 z | 300/900 | 300s +0.0658（+3.06） | 0.406 |
| hidden_bid_support_z_300s | 买方隐藏深度支撑 z | 900 | 900s −0.0763（−4.34） | 0.434 |
| hidden_imb_pos_frac_300s | 隐藏深度失衡的占比形式 | 900 | 900s −0.2051（−2.98） | 0.285 |
| ofi_sign_persist_60s | OFI 符号持续性 | 15/30 | 15s +0.0713（+3.63） | 0.782 |

头部多空（τ=5%，OOS 5 天，仅描述）：本轮有 **2 个因子双边净正**——div_pos_frac_300s（900s 多头 +12.1bp / 空头 +10.9bp）和 dev_open_x_spread_z（900s +2.2 / +25.1）；比第二轮（1 个）多一个。其余头部扣 6bp 后多为负，依旧只作描述。

## 去重（对 35 个库内因子 + 批内，314,053 行 pooled Spearman）

砍 9 个，本轮去重损耗明显高于前两轮（round 1 砍 3、round 2 砍 2）：

对库内近克隆（8 个，规则：保先入库者）：
- R3-A 全部 4 个过筛者：dev_open_z_300s（ρ0.901 对 mid_roll_range_pos_300s）、open_side_hold_bars（0.879 对 dev_from_open_bps）、open_pos_roll_range_300s（−0.921 对 dev_from_open_bps）、stretch_per_day_range（0.923 对 dev_from_open_bps）——**锚家族的任何位置/持续/伸缩变换都已是库内因子的克隆**。
- book_div_z_120s：0.872 对 top5_book_div_z_300s（同背离换窗口）。
- ofi_per_vol_z_300s：0.958 对 ofi_per_depth_z_300s（按成交量归一 ≈ 按深度归一）。
- ofi_flip_fade_300s：−0.952 对 ofi_per_depth_z_300s（符号翻转克隆）。
- top5div_x_extremes：0.930 对 top5_book_div_z_300s（极值门控盖不住基底）。

批内孪生（1 个）：top5div_x_spread_z 与 div_z_x_spread_z **ρ=1.000**——两个子代理独立写出了同一个公式 z(wdi−全簿失衡)×z(价差)；保 R3-B 版本 div_z_x_spread_z。

观察名单（0.65-0.80，记账不砍）：hidden_bid_support_z_300s × conc_imb_z_300s −0.792；div_pos_frac_300s × hidden_imb_pos_frac_300s −0.733（同族兄弟）；dev_open_x_spread_z × range_pos_x_spread_z +0.728（锚不同）；ofi_sign_persist_60s × ofi_concord_15_60 +0.723。

## 死因地图（38 个 FAIL + 9 个去重砍）

| 组 | IS 就死 | OOS 崩 | 面板重复 | 去重砍 |
|---|---|---|---|---|
| R3-A 锚动态 | 10 | — | — | 4 |
| R3-B 深簿背离 | 3（背离×流对齐 2、可见占比 z） | 4（隐藏失衡裸 z 及其交互） | — | 1 |
| R3-C 跨尺度流 | 1 | 8（符号连击/流 t 统计/确认类） | 2（ti15_per_vol_regime 0.949、trade_book_conflict_60s 0.886） | 2 |
| R3-D 状态交互 | 7（×rv_z、×ofi_z、×ti、×wdi 门控全灭） | 1 | 2（rollrpos_x_ti15 0.929、dayedge_x_ti60 0.927） | 2 |

关键死因结论：

1. **锚家族饱和**：R3-A 14 个里 10 个 IS 就死、4 个过筛后全是库内克隆——dev_from_open_bps 与 mid_roll_range_pos_300s 已把"价格相对日锚的位置"信息采完，此方向关停。
2. **状态门控只有价差维活**：R3-D 的 ×rv_z / ×ofi_z / ×ti / ×wdi / ×fbi 门控 8 死，只有 ×spread_z 三个全活（div/dev_open/conc_imb）。报价压力是 588000 上唯一稳定的交互维。
3. **已入库因子的再组配是去重陷阱**：换窗口（book_div_z_120s）、换归一化（ofi_per_vol）、加门控（top5div_x_extremes）都留得住母体信号 → 5/9 的砍单源于此。新构造必须换**经济输入**而不是换变换。
4. 跨尺度流的"确认/连击/t 统计"类 8 个 OOS 崩：快慢流的方向一致性想法大多只是把已有快列信号重新讲一遍。

## 经验沉淀（给第四轮）

1. **spread_z 门控是富矿**：三个 ×spread_z 交互全过（其中 2 个全 horizon），且对面板相关低（0.22-0.42）。库里还有一批没做过 spread-z 交互的存活因子（ofi_concord_15_60、fullbook_imb_mom_60s、ti_accel_15_60、ofi_sign_persist_60s、iopv_vel 系），下一轮优先补。
2. **隐藏深度单边分解刚开张**：买/卖隐藏深度的 z 与占比形式都活（900s 为主），与库内 conc_imb/top5_div 相关仅 0.4-0.6；可做其动态（变化率）、与成交流的方向配合。
3. **ofi_z_cross_vel_15s（z 与瞬时速度的背离，4 horizon 过）**说明"水平状态 vs 瞬时动态"的失配本身是信号；可推广到 wdi/div/premium 等其它状态列。
4. 已饱和关停：锚位置衍生（任何形式）、已入库因子的换窗/换归一/门控再组配、跨尺度"确认/连击"叙事。

## 归档位置

- library/candidates.json 键 `iter003_round3`（门槛/划分/10 明细含头部/9 砍掉/死因分组/经验）。
- scratch-iter003/：reports_round3/（19 份 PASS 报告）、round3_admitted_detail.json、round3_deathmap.json、iter003_round3_corr.json/.txt（服务器副本 /data/factor_lzt/iterations/）、本报告（服务器副本同处）。
- 服务器日志：/data/factor_lzt/logs/iter003_round3.log；脚本 /data/factor_lzt/scripts/iter003_round3.sh、round3_corr.py。
