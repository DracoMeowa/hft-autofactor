# 04 — Look-Ahead 防范与 Point-in-Time 验证

> **更新日期: 2026-08-04** · 状态: 现行
> 来源：上一轮工作流的防泄漏专项调研（五支柱框架）+ 本项目实现/验证中的工程教训。
> 追加规则：新发现的泄漏模式或验证手段追加到 §6/§7；已落地的工程约定标注「已实现」。

本项目标签 horizon 最长 900s、行网格 3s、输入是乱序风险很高的交易所转储 ——
look-ahead bias 是头号死因。本文是防泄漏的唯一权威参考。

---

## 1. 五支柱框架（调研结论）

1. **重叠前瞻标签（≤900s）会跨越任何切分边界** → López de Prado purging（删除标签窗口与
   测试段重叠的训练样本；purge 宽度 = 流水线中用到的**最大** horizon）+ embargo 缓冲；
   day-blocked 切分对日内标签自动 purge，但仍需 embargo，且每个 session 最后 15 分钟不产生
   900s 标签是硬规则。
2. **滚动特征需要显式 availability-time 契约**：backward-only as-of join、严格过去窗口 +
   warm-up 期 NaN、因果归一化、不跨日携带状态、auction-aware session 处理。
3. **核心 MASK VALIDATION 设计 = truncate-and-recompute 前缀同一性**：把确定性引擎在序列
   位置 seq(T) 截断的流上重跑，断言 t<=T 的所有输出与全量运行 **bit 级相等**；用并行截断
   运行与 checkpoint-restore 提效；用严格 FP 标志、整型缩放簿状态、有序归约、per-instrument
   单线程累加使 bit 级比较合法。辅以 fuzz 截断、独立 Python 参考实现（differential test）、
   排序确定性测试、对标签也做同样 mask 测试、以及**必须失败**的 canary 泄漏因子。
4. **对齐**：按交易所 SeqNo/ApplSeqNum 排序，绝不按粗时间戳；SeqNo 缺口与快照-重建簿失配
   作为数据质量旗标**排除区间**而非插值；绝不把快照值回填到更早的事件时间；因子引擎、标签
   构造器、参考实现共用**同一份**跨流 tie 约定文档。
5. **验证方案**：day-blocked purged walk-forward（发现期 anchored、上线跟踪 rolling）、
   IS:OOS ≈ 5:1–10:1、只聚合 OOS 指标、per-fold IC/参数稳定性审计；日数积累后升级到
   CPCV + PBO/Deflated Sharpe。

## 2. Purging 与 Embargo

**Purging（López de Prado, AFML Ch.7）**：删除每个标签窗口 `[t_obs, t_obs+H]` 与任何测试
样本时间跨度重叠的训练样本。需要保存每样本 label end time。**purge 宽度必须等于流水线中
最大 horizon（本项目 900s），不是目标自己的 horizon。**

```
train' = { (t,y) ∈ train : t + H_max < min(test_start) }
```

**Embargo**：吸收 purging 漏掉的序列相关（微观尺度上很强）的缓冲区。大小：数据的 0.5–5%
或 ≥ 一个最大标签时长；walk-forward 中还丢弃测试集前 E 个观测；k-fold 中丢弃紧跟每个测试
块**之后**的训练样本（mlfinlab pct_embargo 语义）。

```
test_effective_start = train_end + purge + E_embargo
```

**Day-blocked purged walk-forward（主方案）**：按交易日切分 —— train days 1..k，test day
k+1（anchored/expanding）做因子选择；rolling ~20 天窗口做上线跟踪。标签不跨收盘 ⇒ 日边界
**自动 purge**。只报告拼接的 OOS 指标；IS:OOS ≈ 5:1–10:1；审计 per-fold IC 与参数稳定性
（乱跳 = 过拟合）。

**已实现**：`eval/splits.py::purged_day_splits`（标签均为日内 ⇒ day blocking 自动 purge
900s horizon；embargo_days≥1 显式保留）+ `is_oos_retention`（retention ≥ 0.5）。

## 3. 标签边缘规则（五 horizons）

- 每个连续 session 的**最后 900s/300s 不产生标签**；标签不得跨越 11:30 午休或 14:57–15:00
  收盘竞价（除非显式按交易时间定义）。
- **Label builder 必须输出 ABSENT，绝不 pad/forward-fill 未来价格。**
- `valid_label(t,H) = (t+H <= session_end_continuous) ∧ no_break_between(t, t+H)`。
- 需要 per-exchange session/auction 日历与快照 TickStatus/auction 旗标。

**已实现**：`labels.cpp` LabelBuilder —— future-only、ABSENT 语义、session 边缘规则、
对稀疏标的"首个 >= t+H 的快照"解析；mask test E 证明之。

## 4. MASK TEST 设计（核心）

### 4.1 MASK TEST A — truncate-and-recompute 前缀同一性

- 在全日流 S 上跑引擎；在 K>=4 个点（warm-up、窗口中段、午休后、尾盘）截断：保留
  `SeqNo <= seq(T_k)` 的最后一个事件；重跑；断言 `O_trunc == O_full|_{t<=T_k}` **bit 级**
  （先 hash 比较，失败再 diff）。
- **按序列位置截断，不按时间戳** —— 这样排序 bug 也会被抓住。读未来事件的因子按构造失败；
  canary 泄漏因子必须失败。
- `assert bits(Engine(S[0..T])) == bits(Engine(S))[t<=T]`
- 需要：确定性引擎；golden 输出 + xxh3/SHA-256 hash 存 `/data/factor_lzt/validation/`。

**已实现**：`validation/mask_test.py`（`hftaf mask`），截断点 warmup/mid_am/post_lunch/late，
golden hash 在 `validation/golden.py`。真实数据（20250701 SSE ch6）4/4 截断点 PASS、
canary 按预期失败。**重要语义修正（1f9eff4）**：对快照稀疏标的，截断运行中某标签可能合法地
"解析不到"（首个 >= t+H 的快照落在截断点之后）→ compare 采用 directional 语义：
**截断端 PRESENT 的标签必须与全量相等；截断端 ABSENT 允许**，而非双侧严格相等。

**重要语义修正（71480f9，2026-08-04）**：`ts == T_cut` 的**单个边界行**的非标签列豁免
bit 级比较（标签列仍按上述 directional 语义比较）。根因：SSE tick 流 **SeqNo 严格单调**
（实测 0 违例）但 TransactTime 存在约 10³/日 的乱序戳；归并在「首个 TransactTime > U 的
tick」处停止，SeqNo 截断使停止点移动，边界快照吸收的 tick 集因此不同 → 边界行的
order-flow 类因子（如 ofi_60s）在全量/截断两次运行间合法地不同。**这不是穿越**：
`ts < T_cut` 的所有行仍 bit 级一致（因果性保证不变），canary 仍必须失败（检出力不变）。
实测：20250701 sse ch5 四个截断点除边界行外逐字节一致（修复前四天全部因边界行因子差异
FAIL）。**2026-08-04 复验通过**：20250701 / 20250815 / 20250930（sse ch5，k=4）三天
4/4 截断点 `identical=True`、`canary_failed_as_required=True`，golden hash 已入库
（`validation/golden/*__sse__ch5.sha256`）。
回归测试：`test_mask_pipeline.py` 的 `test_compare_prefix_boundary_row_*` 两用例
（豁免生效 + 边界行标签仍比较）。

### 4.2 Bit 级相同的浮点契约（使 MASK TEST 合法）

编译 `-fno-fast-math -ffp-contract=off`（禁 FMA 收缩）、不用 `-Ofast`、固定 FTZ/DAZ、
per-instrument 单线程累加、跨 instrument 操作有序归约、避免不确定的 libm 超越函数、
价格/量用整型缩放（tick 单位）、金额 int64。**同二进制 + 同机器 ⇒ bit 级可复现；跨机器
退回 ULP 容差模式。**（已实现：CMakeLists 全局 `-O2 -fno-fast-math -ffp-contract=off`，
价格 int64 milli-CNY；服务器 clean rebuild 已验证 bit 级一致。）

### 4.3 配套测试 B/C/D/E

- **B（fuzz）**：随机截断 + 重复事件幂等 + 同毫秒乱序到达（前缀一致性 property）。
- **C（differential）**：对每因子与慢速独立 NumPy/pandas 参考实现对比（整数精确、浮点紧
  ULP）。⚠ **现状警示**：`reference_factors.py` 目前只有自身单测调用、未接入 pipeline，
  且与引擎语义有分歧（RV 链在快照 gap>6s 时断链、要求 20/100 连续收益 vs 引擎无 gap 检查、
  只要求 ≥80% 名义收益；还查 `BidPrice0` 而真实 SSE 列名是 `BidPrice[0]`，真实数据上会
  全 null）。接线前该测试视为**未启用**，靠 spot-check 补位。
  **2026-08-04 更正**：上述分歧已在任务 #66 修复（列名兼容 `BidPrice[0]`、RV 改引擎语义、
  exchange 行域参数，见文末更新日志）；「未接入 pipeline」仍为真——差分测试仍属
  opt-in，生产验证继续以 mask + spot-check 为主。
- **D（ordering）**：`(TransactTime, SeqNo)` 排序 vs 纯 SeqNo 必须输出相同。
- **E（labels）**：同一截断协议用于 label builder —— t <= T−H 的标签不变，更晚的标签 ABSENT。

### 4.4 Canary 泄漏因子

故意前瞻的因子（`future_mid_15s`、`future_trade_sign`）必须在 mask test 中失败，证明验证器
有检出力。⚠ **foot-gun**：`--factors future_mid_15s` 可以在不带 `--canaries` 的情况下构建
泄漏因子（factory 按名字构造）。已提交配置不可达，但手工调用可能把前瞻列静默写进生产形状的
输出 —— 未来应加 factory/CLI 守卫。

## 5. Availability-time / as-of 契约与对齐

- `factor(t)` 只能用 `availability_time <= t` 的数据。backward-only as-of join，同时间戳时
  严格 `<` tie 规则；滚动窗口右闭、warm-up 全 NaN；归一化统计只用因果 expanding/rolling
  窗口；跨 ETF 横截面操作需 per-instrument staleness 检查；**不跨日携带状态**（channel
  分片逐日变化）。
- **SeqNo-first 排序与 gap gating**：SSE TransactTime 毫秒粒度（每毫秒多事件）、快照 3s
  更新 ⇒ 仅时间戳排序有歧义。tick 流按 per-channel SeqNo（SSE）/ApplSeqNum（SZSE）排序；
  用 TrdBuyNo/TrdSellNo（SZSE Bid/OfferApplSeqNum）匹配成交与 resting order。
  **SeqNo 缺口 ⇒ 消息丢失 ⇒ 打旗；跨旗标区间的因子样本从评估中排除，绝不插值。**
- **快照-事件对齐约定**：快照时间戳 U = U 时刻的簿状态。归并规则：`TransactTime <= U` 的
  tick 先于快照 U 处理；`> U` 的后处理；同毫秒 tie 用**一份文档化约定**（因子引擎、标签
  构造器、参考实现相同）。绝不把快照值回填到更早事件时间（静默 LAB）；forward-fill 只能
  向前且打旗。若快照文件带 sequence/DataTime 字段，优先于时间戳推断。
  （已实现：引擎 merge rule「TransactTime <= U 的所有 tick 先于快照 U」。）
- **Book reconciliation 作为持续验证**：从开盘（竞价后）快照 + tick 重建簿；每个后续 3s
  快照比较重建 top-N 档 vs `Bid/AskPrice[0..9]`；失配 ⇒ resync + 记录；持续失败则作废因子
  区间。另查成交方向一致性（TrdBSFlag）、负价差、竞价期消息；集合竞价（9:15–9:25、
  14:57–15:00）按独立状态处理或从连续簿因子中排除。
  （已实现：snapshot-anchored book + `flags` bit0/bit1 语义。）

## 6. 统计泄漏诊断

- **IC-cliff 测试**：lag-0 IC vs lag-H IC；lag-0 > ~3× lag-1（且量级可观）是未来函数的经典
  签名。`IC_k = corr(factor.shift(k), label)`；若 `IC_0 >> IC_1` 打旗。
- **Label-shuffle 测试**：置换 label↔timestamp 映射，模型 IC 必须坍缩到 ~0。
- canary 因子必须被 mask test + 诊断抓住 —— 证明验证器本身有效。
- （来源：中国量化界"未来函数"检测实践；Kaufman et al. leakage detection。）

## 7. Label-drift 告诫（本项目实测发现）

标签在每标的"首个 >= t+H 的快照"处解析；对快照稀疏的标的（如 501xxx LOF），U−t 可能显著
超过 H。**fwd_* 标签因此不是稀疏标的上的精确 H 秒收益**（已文档化，mask 验证器有意容忍截断
运行中标签合法缺席）。推论：
- 评估稀疏标的时 IC 的有效 horizon 被拉长/混合，horizon 间衰减曲线可能被污染；
- 生产跑数建议按流动性过滤标的，或在 eval 中按快照密度分层报告。

## 8. 验证器反过拟合规则与升级路径

- **不要**为了提高验证结果去调窗口大小、embargo 长度或切分参数 —— 那是 meta-overfitting。
  参数从微观结构逻辑（标签 horizon、ACF 衰减、session 结构）选定后**固定**。
- **CPCV 升级路径**：日数足够后，用 Combinatorial Purged K-Fold（n_splits, n_test_splits；
  paths = C(n,k)）+ 每 path purging/embargo，得到 OOS IC/Sharpe 分布 ⇒ Probability of
  Backtest Overfitting 与 Deflated Sharpe Ratio；因子候选量大（多重检验环境）时尤为重要。
- **已实现的相关件**：eval 层 day 内 rank、Newey-West 有效 n、BHY FDR、deflated Sharpe、
  purged day-blocked walk-forward（embargo≥1 day）；backtest 用 trailing causal z-score
  窗口且 `signal_lag_rows >= 1` 强制（ValueError 保护）。
- 独立对抗性复核（2026-08-03/04）结论：生产路径未发现 look-ahead；唯一前瞻代码是两个
  canary 类（`is_canary()=true`，已提交配置中仅 `--canaries` 可达）。

## 9. 建议的验证流水线顺序（CI）

```
ingestion gate（SeqNo gap、快照 reconciliation、竞价分段）
  → MASK TEST A（truncate-and-recompute，golden hash bit 级）
  → fuzz（B） → differential vs Python reference（C，先修复接线/语义）
  → ordering（D） → label mask test（E） → canary（必须失败）
```

并把验证二进制与编译器旗标钉在仓库里；文档声明 bit 级断言只在同二进制/同机器有效，跨机器用
ULP 容差回退。

## 更新日志

- 2026-08-04: 初版入库（journal 恢复 + 集成/验证阶段的工程教训：directional mask 语义、
  canary foot-gun、differential test 未接线、label-drift）。
- 2026-08-04（集成修复）:
  - §4.4 canary foot-gun **已修复**：`make_registry` 对未带 canaries 标志却点名 canary 的
    请求直接抛 `std::invalid_argument`（`hftaf-engine` 捕获后退出码 2）。mask test 的
    `--canaries` 路径与 `test_determinism` 不受影响；`test_factors` 新增拒绝用例。
  - §4.3 测试C 分歧**已修复**（`reference_factors.py` 与引擎对齐）：列名同时接受括号形式
    `BidPrice[0]`（优先，真实 dump）与旧式 `BidPrice0`；RV 改为引擎语义 —— 时间窗裁剪、
    仅单边快照打断相邻链（快照缺失/时间 gap **不再**断链）、warm-up = elapsed ≥ window
    且窗口内收益数 ≥ 80% 名义值；缺失量能字段按引擎 opt-int 语义解析为 0；新增
    `exchange="sse"|"szse"` 参数复现引擎行域（ETF 代码 + 连续竞价段）。旧 gap>6s 断链、
    要求满窗连续收益的语义作废。单测同步重写（223/223 通过）。
- 2026-08-04（mask 边界行豁免，71480f9）: §4.1 新增第二个语义修正——`ts == T_cut` 的
  单个边界行非标签列豁免 bit 级比较。根因与边界（SeqNo 截断 × 乱序 TransactTime 戳的
  归并顺序效应，非穿越；ts < cut 仍 bit 级一致、canary 检出力不变）见 §4.1 正文。
  回归测试 `test_compare_prefix_boundary_row_*` 两用例入库（全套 245/245 通过）。
