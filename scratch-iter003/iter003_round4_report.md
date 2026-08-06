# iter-003 第四轮评估报告（588000，59 列宽表持续迭代）

日期：2026-08-06　批次：iter003_round4（#147）
前置：第三轮入库 10（库 45）；本轮按 round-3 经验定向挖四个方向。

## 一句话结果

56 个候选 → eval-v2 门槛过 26 → 对 45 因子库做 pooled Spearman 去重砍 2 → **入库 24，库总计 69**。过门率 26/56 是四轮最高；去重仅砍 2（都是批内孪生），无一因子对库 ≥0.85——本批构造基本正交于现有库。

## 批次构成（4 个研究组，各由一个 factor-researcher 子代理产出）

| 组 | 主题 | 个数 | 过筛 | 入库 |
|---|---|---|---|---|
| R4-A | spread-z 门控补全（对库内存活因子补做价差门控） | 14 | 10 | 9 |
| R4-B | 隐藏深度动态化（买/卖隐藏深度的速度、与成交流配合） | 14 | 5 | 5 |
| R4-C | 水平 z 与瞬时速度背离（ofi_z_cross_vel 模板推广） | 14 | 9 | **8（本轮之星）** |
| R4-D | 报价形状动态（gap/slope 的衍生与交互） | 14 | 2 | 2 |

流程耗时：注册 ~15 秒；run 56 个 × 66 天 ≈ 5 分钟（因果探针全过）；screen ≈ 5 分钟。

## 门槛（eval-v2，同前三轮）

RankIC 为主：每个 horizon IS+OOS Newey-West |t| ≥ 2.0；retention ≥ 0.5 且 IS/OOS 同号；对面板列 |Spearman| ≤ 0.85；批内/对库再做 pooled Spearman（阈值 0.85，~307k 行 × 66 天）。成本固定单边 3bp（往返 6bp），头部多空统计仅描述。划分：训练 60 天 / OOS 5 天 + 1 天禁运。

## 入库 24 个（按亮点排序）

### R4-C 速度背离簇（8 个，本轮最大收获）

| 因子 | 含义 | 通过 horizon | 最强 OOS IC（t） | 对面板 max\|ρ\| |
|---|---|---|---|---|
| **wdi_zvel_extreme_15s** | 五档深度失衡 z 的"速度极端值" | **全 5 个** | 15s **+0.1804（+13.87）** | 0.466 |
| **oir_z_cross_vel_15s** | 顶档失衡 z 与其 15s 速度的交叉 | 15/30/60/900 | 15s +0.1662（+15.04） | 0.527 |
| **wdi_z_cross_vel_15s** | 五档深度失衡 z 与速度的交叉 | 15/30/60/900 | 15s +0.1409（+11.46） | 0.445 |
| fullbook_imb_z_cross_vel_15s | 全簿失衡 z 与速度的交叉 | **全 5 个** | 900s +0.0140（+10.48） | 0.343 |
| depth5_imb_zvel_extreme_15s | 五档失衡速度极端值 | 15/900 | 15s +0.0980（+6.63） | 0.293 |
| oir_zvel_div_15s | 顶档失衡 z 与速度的商（背离） | 15 | 15s +0.0836（+9.66） | 0.452 |
| microprice_dev_z_fade_pos_15s | 微价偏离 z 正向时衰减 | 15/30/60 | 15s +0.0650（+5.48） | 0.339 |
| ti60_z_cross_vel_15s | 成交失衡 z 与速度的交叉 | 300 | 300s +0.0071（+2.33） | 0.279 |

> wdi_zvel_extreme_15s 的 15s OOS IC +0.18（t +13.9）是全项目迄今最强短周期 IC（此前最强为 top5_book_div_z_300s 的 0.21，但那是不同基底；同基底速度构造里这是头一份）。模板——"状态列的滚动 z 与其自身 15s 变化率做交叉/背离/极端值"——在 wdi/oir/fullbook/depth5_imb/microprice/ti60 上全部成立。

### R4-A spread-z 门控（9 个）

| 因子 | 含义 | 通过 horizon | 最强 OOS IC（t） | 对面板 max\|ρ\| |
|---|---|---|---|---|
| ofi_concord_x_spread_z | OFI 跨尺度一致性 × 价差 z | 15/30 | 15s −0.0721（−5.91） | 0.521 |
| ti_accel_x_spread_z | 成交加速度 × 价差 z | 15/30/60 | 15s −0.0375（−3.95） | 0.309 |
| ofi_persist_x_spread_z | OFI 符号持续 × 价差 z | 15/30/60 | 15s −0.0587（−3.03） | 0.581 |
| iopv_velz_x_spread_z | IOPV 速度 z × 价差 z | 15 | 15s −0.0396（−3.51） | 0.456 |
| fbi_mom_x_spread_z | 全簿动量 × 价差 z | 15/30 | 15s −0.0705（−4.46） | 0.305 |
| ofi_concord_wide_gate | OFI 一致性 × 宽价差门 | 300 | 300s +0.0169（+3.20） | **0.078** |
| fbi_mom_wide_gate | 全簿动量 × 宽价差门 | 30 | 30s +0.0120（+3.58） | **0.060** |
| ofi_persist_wide_gate | OFI 持续 × 宽价差门 | 15 | 15s +0.0157（+4.79） | **0.096** |
| iopv_drift_wide_gate | IOPV 漂移 × 宽价差门 | 900 | 900s −0.0296（−2.51） | **0.022** |

> spread-z 门控的过门率印证 round-3 结论（价差是唯一稳定的交互维）。但有两点新观察：(a) `_x_spread_z`（乘 z）多是短周期负 IC、强度中等；(b) `_wide_gate`（乘价差水平、非 z）面板 ρ 极低（0.02–0.10，几乎完全正交），但 IC 也小——是"几乎正交的弱信号"。批内已现孪生（ofi_concord_x_spread_z 与其 flip_gate ρ=1.000；与 ofi_persist_x_spread_z ρ=0.826），spread-z 门控接近饱和。

### R4-B 隐藏深度动态（5 个）

| 因子 | 含义 | 通过 horizon | 最强 OOS IC（t） | 对面板 max\|ρ\| |
|---|---|---|---|---|
| hidden_bid_support_z_vel_15s | 买方隐藏深度支撑 z 的 15s 速度 | 15/30/900 | 30s −0.0760（−5.45） | 0.266 |
| vis_share_vel_15s | 可见深度占比的 15s 速度 | 60/900 | 60s +0.0156（+5.64） | 0.103 |
| hidden_ask_supply_z_vel_60s | 卖方隐藏供给 z 的 60s 速度 | 300 | 300s +0.0327（+2.63） | 0.282 |
| hidden_ask_x_sellvol_z | 卖方隐藏供给 × 卖出量 z | 15 | 15s −0.0150（−4.29） | 0.053 |
| hidden_bid_arrival_x_buyvol | 买方隐藏支撑 × 买入量 | 15 | 15s +0.0184（+2.10） | 0.084 |

> 隐藏深度"速度/动态"（5/14）明显弱于 round-3 的隐藏深度"水平 z/占比"（7/14）：慢状态做差分后噪声放大。活下来的多是 15s 速度版本或与成交流方向耦合（hidden_ask_x_sellvol、hidden_bid_arrival_x_buyvol）。

### R4-D 报价形状（2 个）

| 因子 | 含义 | 通过 horizon | 最强 OOS IC（t） | 对面板 max\|ρ\| |
|---|---|---|---|---|
| gap_slope_thinwalk_300s | 薄价差时沿 slope 走 | 15 | 15s −0.0400（−10.27） | 0.153 |
| slope_gated_ofi_300s | slope 作门控的 OFI | 15 | 15s +0.0444（+4.29） | 0.558 |

> R4-D 是本轮最差组（2/14）：gap/slope 的衍生及与 OFI/TI 的确认/吸收类 8 个 IS 就死、2 个 OOS 崩、micro_ti_absorb_300s 直接面板重复（0.900）。报价形状家族在 588000 上基本不增量——关停。

头部多空（τ=5%，OOS 5 天，仅描述）：扣 6bp 往返后，**本轮无双边净正因子**（round-3 有 2 个）。速度背离簇虽 IC 强，但头部分位扣成本后双边多为负，依旧只作描述。

## 去重（对 45 个库内因子 + 批内，~307k 行 pooled Spearman）

砍 2 个，**四轮最低**（round 1 砍 3、round 2 砍 2、round 3 砍 9）：

- **ofi_concord_flip_gate**：ρ=**1.000** 对批内 ofi_concord_x_spread_z——符号翻转门控与价差 z 乘积是同一序列；保 x_spread_z（价差门控版，round-3 验证的活方向）。
- **microprice_dev_zvel_extreme_15s**：ρ=**0.851** 对批内 wdi_zvel_extreme_15s——同为"z 速度极端值"构造，微价偏离与五档深度失衡两个基底经验上塌缩到一起；保 wdi 版（全 5 horizon vs 4，面板 ρ 0.466 vs 0.523）。

**无一因子对库 ≥0.85**——spread-z 门控与 z-速度构造对现有 45 因子库基本正交。这是快迭代面板挖矿仍在发现新经济问题（而非换皮）的强信号。

观察名单（0.71-0.85，记账不砍）：ofi_concord_x_spread_z × ofi_persist_x_spread_z +0.826（两个 OFI 价差门，接近）；ofi_concord_wide_gate × ofi_persist_wide_gate +0.791；wdi_z_cross_vel_15s × oir_z_cross_vel_15s +0.740（wdi vs oir，五档 vs 顶档失衡）；wdi_zvel_extreme × depth5_imb_zvel_extreme +0.738；slope_gated_ofi_300s × ofi_per_depth_z_300s（库）+0.713。

## 死因地图（30 个 FAIL + 2 个去重砍）

| 组 | IS 就死 | OOS 崩 | retention/符号 | 面板重复 | 去重砍 |
|---|---|---|---|---|---|
| R4-A spread 门控 | 4（ti_accel/iopv_velz/iopv_drift 的 wide_gate、iopv_drift_x） | — | — | — | 1 |
| R4-B 隐藏深度动态 | 3 | 3 | 3 | — | — |
| R4-C 速度背离 | 4（spread_z_cross/spread_zvel_extreme/ti60_fade/book_slope） | 1（fullbook_imb_zvel_div） | — | — | 1 |
| R4-D 报价形状 | 8 | 2 | 1（gap_vel） | 1（micro_ti_absorb 0.900） | — |

关键死因结论：

1. **报价形状（gap/slope）家族在 588000 上不增量**：R4-D 12/14 死，只有"薄价差时沿 slope 走"和"slope 作 OFI 门控"两个偏门想法活下来；gap 水平、gap×TI/OFI 确认、slope×TI 吸收全灭。此方向关停。
2. **以 spread 为基底的"spread_z_cross_vel / spread_zvel_extreme"死**（R4-C 2 个 IS 死）：价差既是门控维又是被做 z-速度的状态列时，不再产生新信号——价差只能做"门控"，不能做"被门控的状态"。
3. **隐藏深度做差分（速度）不如做水平 z/占比**：R4-B 9/14 死，多半是 IS 死或 OOS 崩；慢状态差分放大噪声。活法是与成交流耦合或用极短(15s)速度。

## 经验沉淀（给第五轮）

1. **"z 水平 vs 瞬时速度"的失配是本轮金矿，且远未挖完**：8/14 入库，wdi_zvel_extreme_15s 创短周期 IC 新高（+0.18）。模板 = 状态列滚动 z 与其 15s 变化率做交叉/背离/极端值，已验于 wdi/oir/fullbook/depth5_imb/microprice/ti60。下一轮优先：把模板用到**还没做过的状态列**（book_slope、iopv_premium、quoted_spread 的速度极端值、rv 的速度），以及"极端值"门控的其它阈值/方向变体。
2. **spread-z 门控接近饱和**：10/14 过门但批内孪生已现（ρ 1.000、0.826）。库里还没做 spread-z 交互的基底已不多；若再做，只挑全新基底，且优先 `_wide_gate`（乘价差水平，面板 ρ 极低 0.02–0.10，正交性强）而非 `_x_spread_z`（已密）。
3. **隐藏深度优先水平/占比，差分要配成交流**：速度版本噪声大；要动起来就与 buy_vol/sell_vol/arrival 同方向耦合。
4. **关停清单**：报价形状（gap/slope）衍生与交互、价差自身做 z-速度状态、慢状态的裸差分。
5. **去重极轻**说明快迭代仍在产出正交新因子，而非换皮——继续。

## 归档位置

- library/candidates.json 键 `iter003_round4`（门槛/划分/24 明细含头部/2 砍掉/死因分组/经验）。
- scratch-iter003/：reports_round4/（26 份 PASS 报告）、round4_admitted_detail.json、round4_deathmap.json、iter003_round4_corr.json/.txt（服务器副本 /data/factor_lzt/iterations/）、本报告（服务器副本同处）。
- 服务器日志：/data/factor_lzt/logs/iter003_round4.log、round4_corr.log；脚本 /data/factor_lzt/scripts/iter003_round4.sh、round4_corr.py。
