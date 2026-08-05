# 18 个存量因子 eval-v2 全量复筛报告（2026-08-05）

## 一、为什么要复筛

两件事让 8 月 5 日之前的所有评测数字作废：

1. **#141 修了切分 bug**：`purged_day_splits` 最后一个 fold 的残留块处理有错，旧的 OOS（测试集）组成是错的。
2. **评测标准换成 v2**：以 IC 为主，旧的 Harvey-Liu 门槛式 gate 废除。

所以 18 个已注册因子全部从零重跑。本报告 + `library/candidates.json` 里的
`eval_v2_rescreen_2026_08_05` 块是今后唯一有效的基线，旧数字一律不要再引用。

## 二、新门槛（v2）

| 项目 | 设定 |
|---|---|
| 通过线 | IS \|t\| ≥ 2.0 且 OOS \|t\| ≥ 2.0 且 retention ≥ 0.5 |
| retention 定义 | \|OOS IC\| / \|IS IC\|，**且要求 IS/OOS 同号**；符号翻转直接不及格 |
| 去重 | 池化 Spearman \|ρ\| ≤ 0.85 |
| 切分 | 锚定 walk-forward 最后一折：训练 60 天（20250701–20250922），测试 5 天（20250924–20250930），embargo 1 天 |
| 面板 | 314,053 行（66 个交易日，588000） |
| 成本 | 固定单边 3bp（往返 6bp），只用于头部统计的 net 列，不进门槛 |
| horizons | 15 / 30 / 60 / 300 / 900s 全测，一个不删 |

## 三、结果：18 进 6

| 因子 | 通过 horizons | 一句话 |
|---|---|---|
| flow_divergence_300s | 15 / 30 / 60 | 冠军；短 horizon 三连过，OOS 比 IS 还强 |
| depth5_delta_60s | 900 | 信号强但衰减快；iter-001 的 15s 入库**撤销**（retention 0.470 差一点） |
| wdi_mom_90s | 900 | 15-60s OOS t 高达 8.5 但 retention 只有 0.38-0.46，不过 |
| dd_flow_300s | 900 | 负向延续因子，头部价差与 IC 方向一致 |
| rv_asym_300s | 900 | 负号因子恢复（v2 不看方向）；与 ofi_accum 相关 0.75 |
| session_clock | 60 | 压线过（15/30s OOS t 只差 0.02-0.04）；当条件变量用，不单干 |

### 通过因子关键数字

**flow_divergence_300s**（15/30/60s，IS→OOS）
- 15s：IC 0.0268（t 3.80）→ 0.0528（t 6.30），retention 1.97
- 30s：IC 0.0231（t 3.31）→ 0.0552（t 4.78），retention 2.39
- 60s：IC 0.0171（t 2.16）→ 0.0581（t 2.49），retention 3.39
- 300/900s 不过（IS 就弱）。头部毛收益 ≤1.5bp，扣 6bp 往返为负——价值在排序/条件化，不在直接吃头。

**depth5_delta_60s**（只有 900s 过）
- 15s：IS IC 0.2010（t 15.8）→ OOS 0.0945（t 10.55），但 retention 0.470 < 0.5，差一丝不过
- 30/60/300s 同样 retention 0.32-0.40 不过；900s retention 0.588、OOS t 2.79，通过

**wdi_mom_90s**（只有 900s 过）
- 15/30/60s OOS t 8.5/7.2/8.5，retention 只有 0.38/0.41/0.46 → 不过
- 900s：IS 0.0466（t 3.37）→ OOS 0.0369（t 2.46），retention 0.79，通过

**dd_flow_300s**（负向，只有 900s 过）
- 900s：IS −0.0488（t −2.27）→ OOS −0.1074（t −2.74），retention 2.20
- 头部：多头 +21.6/+22.1/+14.8bp，空头 −31.4/−17.8/−10.3bp（τ1/5/10%），与负 IC 一致（两条腿同号是 OOS 那周有漂移，不是异常）

**rv_asym_300s**（负向，只有 900s 过）
- 900s：IS −0.0740（t −4.11）→ OOS −0.1283（t −2.17），retention 1.73
- 其余 horizon IS 就弱

**session_clock**（只有 60s 压线过）
- 60s：IS −0.0221（t −2.39）→ OOS −0.0545（t −2.01），retention 2.47
- 15/30s OOS t −1.96/−1.98，只差 0.02-0.04
- 头部 60s 空头腿 −16.6/−9.1/−1.5bp 由开盘时段贡献

## 四、12 个不过的：死因分组

**IS 就死（|t| 全 horizon < 2，OOS 再好也不能收）**
- depth_resiliency：15s IS t 只有 1.60（OOS t 3.31 白搭）
- prem_x_ofi：IS |t| ≤ 1.23；OOS 60s t 2.61 无法采信
- prem_x_wdi：IS |t| ≤ 0.57；OOS 15/30s t −2.71/−2.43 无法采信
- trade_arrival_burst：IC≈0，|t| ≤ 1.18
- large_trade_share_level：|t| ≤ 1.90 全灭（rejected_duplicate 只是结构性重复标记，真正死因是 IC）

**IS 强、OOS 崩（regime 变了，不是过拟合噪声）**
- iopv_premium_mom：IS t −4.4~−7.0 → OOS 全 ≈0。溢价动量 regime 在测试周之前就结束了
- queue_refill_asym_300s：IS IC ~0.10（t 6.38）→ retention 崩到 0.08-0.20
- ti_ewm_accel_120s：IS t 4.5/3.9/5.2/3.8/2.9，OOS 全 horizon 反号，|t| ≤ 1.39
- vol_adj_slope：IS t −2.1~−2.3，OOS ≈0，900s 反号

**差一点**
- ofi_accum_300s：15/30s 反号；900s OOS t −1.53（差 0.47）；且与通过的 rv_asym 相关 0.75，重复
- ti_accum_300s：15s retention 0.4997（差 0.0003）且 OOS t 只有 0.51；900s OOS t −1.13
- ti_ewm_state_300s：900s OOS t −1.39（差 0.61）

## 五、去重检查

- PASS6 内部最大相关 0.503（dd_flow × rv_asym），全部低于 0.85 门槛
- **簇警告**：dd_flow × ti_accum ρ=0.894、dd_flow × ti_ewm_state ρ=0.859、rv_asym × ofi_accum ρ=0.750。同簇因子今后一批只留一个代表
- session_clock 对一切行情因子最大 ρ 只有 0.252 —— 正交的时间维度，适合做条件变量

## 六、头部多空统计怎么读（约定，写死）

1. 只在 OOS 5 天上算
2. 多头头 = 每天因子值 ≥ q(1−τ) 的行；空头头 = ≤ q(τ) 的行
3. **short_gross 是底部腿均值取了负号**：判断与 IC 方向是否一致，比较 long_gross 和 −short_gross（底部原始均值）
4. net = gross − 6bp 往返
5. 两腿同号 = OOS 那周整体漂移或时段集中，不是 bug
6. τ=1% 每天只有约 40 行，长 horizon 基本是噪声（例：wdi_mom_90s IC 为正，τ1%/5% 腿看起来反了，只有 τ10% 一致）
7. 只做描述，永不进门槛

## 七、新旧裁决对照

| 因子 | 旧裁决 | 新裁决 |
|---|---|---|
| flow_divergence_300s | admitted@15 | **确认并扩大到 15/30/60** |
| depth5_delta_60s | admitted@15 | **15s 撤销**，只有 900s 过 |
| rv_asym_300s | rejected_sign_gated | PASS@900（v2 不看符号方向） |
| dd_flow_300s | rejected | PASS@900 |
| wdi_mom_90s | rejected | PASS@900 |
| session_clock | rejected | PASS@60（压线，监控+条件用） |
| iopv_premium_mom | rejected_sign_gated | 仍不过；真因是 regime 断裂，不是符号门槛 |
| ofi_accum_300s | rejected_sign_gated | 仍不过（900s 差一点 + 与 rv_asym 重复） |
| 其余 10 个 | rejected | 确认不过 |

## 八、经验教训（已同步写入记忆）

1. **裁决只对产生它的 eval 代码版本有效**。切分/IC 计算一旦修 bug，必须全量复筛，旧数字全部作废。
2. **retention ≥ 0.5 是短 horizon 的头号杀手**。depth5_delta / wdi_mom / queue_refill 在 15-60s OOS |t| 3-11 却 retention 0.08-0.47：是 regime 依赖，不是死信号；换窗口/归一化后 horizon 会迁移（前两者在 900s 活）。不要原样克隆换窗口重提。
3. **900s 是目前最肥的 horizon**（6 个通过里占 4 个）；15-60s 目前只靠 flow_divergence。任何 horizon 都不许预先放弃。
4. 头部统计读法见第六节（short_gross 带负号；τ1% 是噪声；同号腿是漂移）。
5. 流累加器簇（dd_flow/ti_accum/ti_ewm_state，ρ 0.86-0.89；rv_asym×ofi_accum 0.75）一批只留一个代表。
6. session_clock 是正交时间维度，做条件化/去趋势的工具，不是独立可交易因子。
7. **polars 1.43 坑**：`ewm_mean` 遇到前导 null（引擎 warm-up）会把状态永久污染成 NaN（两种 ignore_nulls 模式都中招），一律改用 `rolling_mean`。
8. iopv 溢价动量是 regime 内的东西：IS 强 OOS 零。溢价类因子必须先加 regime 条件化再提交（iter-003 family D 就是干这个）。

## 九、下一步

1. iter-003 首批 60 个 spec 已写好、冒烟全绿 → 整合、查重名、注册、上服务器批量跑
2. 宽表扩容（#144）：C++ 引擎加 ~17 列（盘口总量、开高低收、iopv 速度、15s/30s 流变量等），只需 stage2 replay + stage3 转换，约半小时
3. 继续「生成一批 → 评估 → 归档 → 写经验 → 下一批」的循环

## 附：产物清单

| 文件 | 内容 |
|---|---|
| `scratch-iter003/rescreen_reports.json` | 18×5 原始报告 JSON |
| `scratch-iter003/rescreen_summary.txt` | 压缩摘要（本报告数字来源） |
| `scratch-iter003/rescreen_corr_out.txt` | 池化 Spearman 相关矩阵 |
| `scratch-iter003/rescreen_block.json` | 归档块（已并入 candidates.json） |
| `library/candidates.json` → `eval_v2_rescreen_2026_08_05` | 机器可读归档 |
