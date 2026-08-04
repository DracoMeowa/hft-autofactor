# 05 — 因子选择与统计门控（阈值 · 多重检验 · Ledger 治理）

> **更新日期: 2026-08-04** · 状态: 现行
> 来源：上一轮工作流的因子筛选/多重检验专项调研（journal 恢复）+ 本项目 `eval/gating.py`、
> `eval/splits.py`、`eval/ic.py` 的已实现参数。追加规则：新阈值校准、新门控追加到对应章节
> 末尾并在 §11 更新日志登记；阈值变更必须给出依据（trial 数变化 / 数据积累 / 成本 regime 变化）。

自动挖掘的 factor zoo 中，**每一个被评估过的变体都是一次 trial**。所有统计校正只有在
诚实的 N 下才有效 —— 这是本文一切阈值的出发点。

---

## 1. 总体门控流程（推荐工作流，调研综合）

```
Stage 1  IS 筛选（train+val，purged）
   多重检验校正后的 RankIC/ICIR/t 值门槛 + 置换噪声底 + BHY-FDR + DSR
      │  全程强制 TrialLedger 记账（先记账，后读阈值）
      ▼
Stage 2  Pristine OOS 确认（每个因子恰好一次）
   OOS/IS retention ≥ 0.5–0.7、OOS t ≥ 2.0、符号一致、win rate、
   跨 horizon 平滑衰减、regime 稳健
      ▼
Stage 3  成本门控（任何 paper trading 之前）
   三档佣金情景全存活；net IR > 0；break-even IC 过关；
   +1–3s 执行滞后下 IC 仍达标；半衰期 > 执行时延；T+1 结算约束满足
      ▼
Stage 4  入库（dedup + 正交化）
   对 library 的 max |Spearman ρ| ≤ 0.7；HAC 聚类代表；
   残差（正交化后）信号通过缩减门槛；批次级 SPA 检验
      ▼
Post-admission  持续监控与退役
   rolling RankIC/EWMA vs 准入基线；黄旗/红旗/退役三级规则；季度衰减重估
```

## 2. 主指标：RankIC、ICIR 与有效样本量

- **Spearman Rank IC 为主指标**：逐 snapshot timestamp 横截面（across ETFs）计算，再聚合
  IC 时间序列。tick/短 horizon 收益非正态、有极端离群值，RankIC 0.04 通常优于被尾部驱动的
  Pearson IC 0.06。警惕微观结构污染（bid-ask bounce、queue 效应）抬高表观 IC。
- **ICIR = mean(IC)/std(IC) 比 IC 水平更重要**：IR>2 优秀，1–2 可用，<1 扣成本后大概率
  不可部署。
- **t 值必须用有效样本量**：3s 快照行 + 最长 900s 重叠标签 ⇒ 标签重叠 ~300 倍、序列相关
  极强，naive t 会高估一个数量级。用 Newey-West / Lo (2002) 调整：
  `t = mean(IC)/std(IC) · sqrt(n_eff)`，`n_eff ≈ n / (1 + 2·Σ_k ρ_k)`。
- **现实 HF 接受域**：日内/分钟级 mean raw IC 0.01–0.03（RankIC 0.02–0.05）属正常；
  中国 A 股 HF 实践常要求分钟级因子 mean RankIC ≥ 0.03–0.05 且 ICIR ≥ 1.0–1.5（滑点吃掉
  弱信号）。**阈值应按 horizon 用置换噪声底校准，而非普适常数。**

**已实现**：`eval/ic.py`（scipy-free Spearman、`newey_west_n_eff`、time-series 与
cross-section 两种 IC 口径）；`GateConfig.min_rank_ic = {15/30/60s: 0.02, 300/900s: 0.03}`、
`min_icir = 0.5`（v1 保守起步值，见 §9）。

## 3. 多重检验栈（zoo 越大越严）

### 3.1 t-hurdle 随 trial 数抬升
- Harvey-Liu-Zhu (2016, JoF 71(1))：~316 个已发表因子下 t>2.0 不够，新因子需 **t > 3.0**。
- 独立 trial 下零假设最大 t 统计量期望 ≈ `sqrt(2·ln N)`（Gumbel/EVT）：
  N=20→~2.4，N=100→~3.0，N=400→~3.5，N=1,000→~3.7，N=10,000→~4.3。
  试验间相关时 `sqrt(2 ln N)` 是上界；用 N_eff（谱分解/聚类计数估计有效维数）。
- **已实现**：`gating.t_hurdle(n) = max(3.0, sqrt(2·ln max(n,2)))`，n 来自 TrialLedger。

### 3.2 FDR：BHY（任意依赖）
- Bonferroni 对 zoo 筛查太保守。BH 在独立/PRDS 下控制 FDR；挖掘因子彼此相关，用
  **Benjamini-Yekutieli**：临界值 `p_(k) ≤ k/(m·c(m))·q`，`c(m)=Σ_{i=1..m} 1/i ≈ ln m + 0.577`。
  m=300 时约严 6 倍 —— 用**先去重（聚类代表）再做 FDR** 找回功效；高级选项是
  factor-adjusted FDR / knockoffs。
- **已实现**：`gating.bhy_critical_values(m, q=0.10)`；`GateConfig.fdr_q = 0.10`。

### 3.3 Deflated Sharpe Ratio 与 MinTRL
- DSR = 观测 SR 胜过 N 次 trial 零假设最大 SR 期望的概率（考虑偏度/峰度）；要求
  **DSR ≥ 0.95（p < 0.05）**。N=100 时把 SR=1 与运气区分开可能需要几十年数据 ——
  HF 场景靠横截面（多标的短窗口）而非单一长记录；MinTRL 给显著性所需最短记录长度。
- **已实现**：`gating.deflated_sharpe_pvalue(sr, n_trials, T, skew, kurt)`。

### 3.4 Harvey-Liu Sharpe haircut
- 报告 SR/t 按 f(N, ρ̄, T, α) 打折：基准例 N=100、ρ̄=0.3 → 有效独立试验 ~20–30，要求 t
  从 2.0 升到 ~2.8–3.0；10 年 raw SR 2.0（N=50, ρ̄=0.3）折后 ~1.2–1.4。准入时存储 haircut
  进因子 metadata。

### 3.5 批次级 family-wise：Hansen SPA
- White Reality Check 对全部候选规则 bootstrap 最大统计量，但纳入大量弱候选时保守/低功效；
  **Hansen (2005) SPA** studentize + re-center，size 正确且功效更高 —— 每个挖掘批次做
  「最佳挖掘因子 vs 基准（零 IC 或当前 library blend）」检验，用 stationary/block bootstrap
  （Politis-Romano）保留依赖。Python 可在 arch / R rugarch 找实现。
- **现状**：本项目尚未实现（v2 项）。

### 3.6 增量价值检验：double-selection LASSO
- Feng-Giglio-Xiu (2020, JoF, "Taming the Factor Zoo")：(1) LASSO 选预测收益的因子，
  (2) LASSO 选预测候选因子的因子，取并集后做 post-selection inference —— 得到候选因子
  增量价值的有效 t 值。是 §6 正交化准入门槛的正式计量版本。

### 3.7 置换/安慰剂噪声底
- 流水线自身的筛选偏差难以解析建模：把 label-shuffle（或符号翻转）的 null 因子走完全相同
  的挖掘+筛选流水线，测每个 horizon 上主指标的 null 分布；IS 阈值取如 null 的 99.9 分位，
  或要求真实因子超出若干 margin。匹配自相关/波动聚集的合成数据同理。
- **已实现**：`gating.permutation_noise_floor(panel, factor, horizon_s, n_perms=50)`；
  `GateConfig.noise_floor_pct = 99.9`。

## 4. 切分设计（非平稳序列）

- **主切分**：chronological train (~60%) / validation (~20%) / pristine test (~20%，最近
  N 个月，每个因子只用一次)。所有边界 purge = 最大标签 horizon（900s）+ embargo ≥ 一个
  horizon。
- **Walk-forward**：非平稳下优先 rolling（固定）窗口而非纯 expanding —— 旧 regime 污染；
  可用 break detection（CUSUM/Bai-Perron）自适应窗口。**fold 数本身也是多重检验**，计入 DSR。
- **Day-blocked purged walk-forward（本项目主方案）**：日内标签 ⇒ 日边界自动 purge 900s；
  embargo_days≥1 显式保留；anchored 发现期 / rolling 上线跟踪。
- **CPCV**：N=10–15 组、k=2（≥45–100 条路径），得 OOS 分布与 PBO。PBO≈0.5 ⇒ IS/OOS 排名
  不相关，>0.9 严重过拟合，<0.3 保守 —— 组合/模型类因子门控 **PBO < 0.5（理想 <0.3）**。
- **Pristine holdout 只用一次**：每次重测消耗统计功效，必须计入 trial ledger；维护
  paper-traded shadow set 分离因子衰减与 regime 效应。

**已实现**：`eval/splits.py::purged_day_splits(dates, n_test_days=5, mode=anchored|rolling,
embargo_days=1)`。CPCV/PBO 为升级路径（未实现）。

## 5. 预期衰减与 retention 门

- **McLean-Pontiff (2016, JoF)**：97 个学术异象 IS→OOS 衰减 ~32%（发表前，统计偏差），
  发表后 ~58%（学习+套利）；高成本异象衰减更慢（成本阻挡套利）。
- **门控**：要求 **OOS/IS retention ≥ 0.5–0.7**（本项目取 0.5），并预算入库后仍持续衰减。

**已实现**：`splits.is_oos_retention(is_ic, oos_ic, min_retention=0.5)` +
`gating.stage2_oos_gate`（retention、符号、`min_oos_t=2.0`、win rate ≥ 0.55）。

## 6. Dedup 与正交化准入

- zoo 中原始相关 0.7–0.9+ 常见，多数因子无增量功效（Freyberger-Neuhierl-Weber 2020；
  Chen-Zimmermann 2022 Open Source Asset Pricing）。
- **流程**：对距离 `1 − |Spearman ρ|` 做层次聚类（HAC）；候选对 library 任一因子/聚类
  `max |ρ| > 0.7` → 拒绝或合并；每簇留一个代表，按 OOS ICIR、换手、容量选择。
- **边界情形**：残差化（有序准入用 Gram-Schmidt，否则对称正交化），要求残差仍过缩减门槛
  （残差 t ≥ ~2.5、显著增量 R² / GRS 型 spanning test, Gibbons-Ross-Shanken 1989）。
  PCA 去重牺牲可解释性；double-selection LASSO（§3.6）是正式计量版本。
- **现状**：本项目 v1 library 尚小，Stage 4 未实现；因子数量上来前必须落地。

## 7. 成本感知筛选（详见 [03](./03-ashare-etf-costs.md)）

- Gross alpha ≠ net alpha：高换手发表异象大部分被成本杀死（Novy-Marx-Velikov 2016 RFS）。
  目标函数内嵌换手惩罚 `λ·turnover`，或在 gross-return vs turnover Pareto 前沿上排序。
- **Break-even IC**：`IC_be ≈ round_trip_cost / σ_dispersion`；等价地
  `net IR = IC·√breadth − cost·turnover/σ` 须 > ~0.5–1.0（含成本安全边际 1.5–2x）。
  本项目往返 8–12bp（流动档）、前瞻离散度 ~50bp 时 IC_be ≈ 0.0016–0.0024 —— 但 ETF 宇宙小、
  breadth 低，必须逐因子直接算 net IR。
- HF 成本非线性、状态依赖：用 tick/depth 成本估计 + 安全边际，不用平 bps。
- **Tradability gate**：信号若 5s 衰完而执行时延+排队 8s 则一文不值。把因子前移
  +1–3s（+1 snapshot）重测 IC 并要求仍过关；测每 horizon IC 半衰期，拒绝半衰期 <
  latency + 决策周期的因子。
- **已实现**：backtest `signal_lag_rows ≥ 1` 强制；`gate_on_costs` 要求三档佣金情景全存活
  （`min_net_sharpe=0.5`、`min_days=20`）。

## 8. 入库后监控与退役

- **监控指标**：rolling 20/60 交易日 RankIC + EWMA vs 准入基线、ICIR 漂移、hit-rate 漂移、
  turnover creep（衰减信号常靠抬换手维持收益）、crowding 代理。平滑单调的 IC-vs-lag 衰减
  曲线 = 真信号；乱跳 = curve-fitting。
- **黄旗**：rolling IC < 准入值 ~50% 连续 2 个窗口，或 rolling Sharpe 低于预期 → 复核降权。
- **红旗**：净 PnL 归因显著负漂移或 regime break → 减仓/停用。
- **退役**：net-of-cost alpha 在回看期为负（低频 6–12 个月；HF 压缩到 **~60–90 交易日**），
  或对 library 级 Sharpe 边际贡献 ≤0；max drawdown / 滑点超模型 2σ 触发自动 kill switch。
  季度衰减重估（半衰期 < 执行时延即退）。保留 retired 因子的 paper-traded shadow 作对照。
- **现状**：未实现（生产跑数积累后落地）。

## 9. 已实现门控清单（eval/gating.py + splits.py，代码为准）

| 项 | 实现 | 默认值 |
|---|---|---|
| min_rank_ic | `GateConfig.min_rank_ic` | 15/30/60s: 0.02；300/900s: 0.03 |
| min_icir | `GateConfig.min_icir` | 0.5 |
| t-hurdle | `t_hurdle(n_trials_eff)` | max(3.0, √(2·ln N))，N 来自 ledger |
| BHY FDR | `bhy_critical_values` + `stage1_screen` | q ≤ 0.10 |
| 置换噪声底 | `permutation_noise_floor` | 99.9 分位，n_perms=50 |
| Deflated Sharpe | `deflated_sharpe_pvalue` | p ≤ 0.05（DSR ≥ 0.95） |
| TrialLedger | append-only JSONL（reports/trial_ledger.jsonl） | **先记账后读阈值** |
| purged splits | `purged_day_splits` | n_test_days=5，embargo_days=1，anchored/rolling |
| OOS retention | `stage2_oos_gate` | retention ≥ 0.5，OOS t ≥ 2.0，win rate ≥ 0.55 |

**调研阈值 vs 实现的差异**（诚实记录）：
- 调研建议分钟级 ICIR ≥ 1.0–1.5；实现取 0.5 起步（单日 smoke 的 IC 偏高不可作为校准依据，
  待多日生产数据后按 horizon 用噪声底重校）。
- SPA、CPCV/PBO、dedup/正交化、衰减监控均未实现（§3.5/§4/§6/§8 标注）。

## 10. 治理规则（不可妥协）

1. **TrialLedger 诚实记账**：每个被评估变体（feature × operator × params × horizon × date，
   含失败分支与人工重试）自动入账；DSR/haircut/√(2 ln N) 全部由它驱动；阈值随 ln(N) 自动
   收紧 —— **不允许事后估计 N**。
2. **Pristine 段重测必须扣账**：没有 ledger debit 不得在 pristine 段重测；pristine 窗口
   用尽后向前滚动（旧测试段转 train，新近窗口成为 pristine）。
3. **每季度复审门控校准**：zoo 规模、成本 regime、市场微观结构都在变。
4. **反过拟合**：不以任何方式调切分/embargo/窗口参数去最大化验证结果（meta-overfitting，
   见 [04](./04-lookahead-prevention.md) §8）。
5. **避免清单**：盲符号回归单独准入（项目要求经济学根基）、对原始相关因子直接 Bonferroni
   （用 BHY 或 cluster-then-FDR）、固定 bps 价差模型（用 ticks）、expanding-only 验证、
   不随 trial 数缩放的固定阈值。

## 11. 主要文献

- Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Journal of Finance*, 71(1). doi:10.1111/jofi.12368
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1).
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014/2017). The Probability of Backtest Overfitting. SSRN 2326253.
- Benjamini, Y., & Hochberg, Y. (1995); Benjamini, Y., & Yekutieli, D. (2001). *Annals of Statistics*.
- Bajgrowicz, P., & Scaillet, O. (2014). *Journal of Financial Economics*.
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*; Hansen, P. R. (2005). *JBES*; Politis & Romano (1994) stationary bootstrap.
- Feng, G., Giglio, S., & Xiu, D. (2020). Taming the Factor Zoo. *Journal of Finance*.
- McLean, R. D., & Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance*, 71(1).
- Novy-Marx, R., & Velikov, M. (2016). A Taxonomy of Anomalies and Their Trading Costs. *RFS*, 29(1).
- Freyberger, J., Neuhierl, A., & Weber, M. (2020); Chen, Z., & Zimmermann, T. (2022).
- Gibbons, M. R., Ross, S. A., & Shanken, J. (1989). *Econometrica*.
- Jensen, T. I., Kelly, B., & Pedersen, L. H. (2023). Is There a Replication Crisis in Finance?
- Grinold, R., & Kahn, R. *Active Portfolio Management*（fundamental law, breadth/IC/IR）; Lo, A. (2002). The Statistics of Sharpe Ratios.
- Pardo, R. *The Evaluation and Optimization of Trading Strategies*; López de Prado, M. (2018). *Advances in Financial ML*（Ch.7 purging/embargo/CPCV, Ch.12）.

## 更新日志

- 2026-08-04: 初版入库（journal 调研恢复 + 已实现 GateConfig 对照）。实现与研究差异已在
  §9 明示；CPCV/SPA/dedup/监控为已规划未实现项。
