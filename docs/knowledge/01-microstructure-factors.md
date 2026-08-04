# 01 — 微观结构因子手册（公式 · 经济学 · 衰减证据）

> **更新日期: 2026-08-04** · 状态: 现行
> 本文合并自原 `docs/microstructure_factors.md`（2026-08-04 前的完整调研内容均已并入）。
> 追加规则：新因子家族请追加到 §14 之后并同步更新 §15 总表与 §17 更新日志；
> 修正既有结论时保留原文并标注 `[DEPRECATED]`。

**范围**：有经济学根基、可在 Level-2 逐笔 + 3 秒快照数据上计算、对短 horizon
（15s–15min）有预测力的微观结构因子。
**预测 horizons**: 15s / 30s / 60s / 300s(5min) / 900s(15min)。
**资产范围 (v1)**: 仅 ETF（上交所 50/51/52/56/58xxxx，深交所 15/16xxxx）。

> 数据约定：tick 文件 `1_channel_N.csv.gz` 为逐笔委托/成交流（20 列，含
> `Trade2_Order1`, `Price`, `Volume`, `OrdSide`, `TrdBSFlag`, `TransactTime`）；
> 快照文件 `1_snapshot.csv.gz` 为 3 秒十档簿（约 180 列，含
> `BidPrice[0..9]/BidVolume[0..9]/AskPrice[0..9]/AskVolume[0..9]`、
> `BidNumOrders[0..9]/AskNumOrders[0..9]`、`TotalBidVolume/TotalAskVolume`、
> `LastPrice`、`IOPV`）。ETF 二级市场 T+0 类别与 T+1 规则见
> [02](./02-etf-microstructure.md)/[03](./03-ashare-etf-costs.md)，tick = 0.001 元。

---

## 0. 执行摘要与可行性地图

现代微观结构文献中最稳健的发现：**同期 order-flow imbalance (OFI) 线性驱动 mid 价格变化**，
在秒级 horizon 上 R² 高、随 horizon 拉长迅速衰减；depth/queue imbalance 是其"潜在供需"
对应物，预测*下一次*价格变动。这两族 —— *已实现流*（成交/委托）与 *潜在流*（簿深度）——
是我们的 horizons 上最强的方向性信号。流动性/波动率家族（Amihud、Kyle lambda、价差分解、
realized vol）最适合做**条件/缩放变量与风险（方差）预测**，而非独立方向性因子。VPIN 与
duration/queue-position 更多是 regime/波动率指标，而非秒级方向 alpha。

**snapshot-only（3s 簿）vs full tick-stream 可行性：**

| 因子家族 | 仅 3s 快照 | 全逐笔流 |
|---|---|---|
| Depth / order-book imbalance（最优档与多档） | **可行（首选）** | 可行 |
| Order-book slope / shape | **可行（首选）** | 可行 |
| Microprice / imbalanced price | **可行（首选）** | 可行 |
| Bid-ask spread（quoted） | **可行（首选）** | 可行 |
| Quote-based realized vol / kernel / semi-variance | **可行（首选）** | 可行（tick-time 变体） |
| Amihud illiquidity | **可行**（ret/volume 分箱） | 可行（更细） |
| ETF IOPV premium/discount | **可行（首选）** | n/a |
| Trade / signed-volume imbalance | 近似（对 `LastPrice` 用 Lee-Ready） | **精确（`TrdBSFlag`）** |
| Cont–Kukanov–Stoikov OFI | 近似（最优档净变化；混淆成交与撤单） | **精确** |
| Order arrival & cancel rates | 不可（只有净值） | **可行** |
| Kyle lambda（回归） | 近似 | **可行** |
| Spread decomposition (Roll / Glosten–Harris) | 近似（对 `LastPrice` 用 Roll） | **可行** |
| VPIN | 近似（对 `LastPrice` 用 BVC） | **可行** |
| Trade duration / ACD | **不可** | **可行** |
| Queue position / queue-reactive | **不可** | **可行** |
| Resiliency / 大单后深度回补 | 不可 | **可行** |

经验法则：**depth/shape/价格位置类特征从快照白拿；凡需要事件排序、成交主动方、撤单、
duration 的，必须全逐笔流。** 快照足以原型化 book-shape、microprice、波动率与 ETF premium
因子；流因子与队列因子强烈建议用逐笔流。

---

## 1. Order Flow Imbalance (OFI) — Cont, Kukanov & Stoikov

**公式**：OFI 把所有最优档事件聚合成净买压。设事件/时刻 `n` 的最优买卖价量为
`(P^b, q^b)`, `(P^a, q^a)`：

```
e^B_n = 1{P^b_n >= P^b_{n-1}} * q^b_n  -  1{P^b_n <= P^b_{n-1}} * q^b_{n-1}
e^A_n = 1{P^a_n <= P^a_{n-1}} * q^a_n  -  1{P^a_n >= P^a_{n-1}} * q^a_{n-1}
OFI_n = e^B_n - e^A_n
```

核心实证关系是**线性价格冲击**：`ΔP_t = β · OFI_t + ε_t`。

**所需 L2 字段**：全逐笔流 —— 每个委托/成交事件上的最优买卖价量
（`1_channel_N.csv.gz`: `Price`, `Volume`, `OrdSide`, `Trade2_Order1`, `TransactTime`）。
快照只能给*粗粒度* OFI（3s 帧间最优档量净变化），会把市价单与限价挂撤混在一起。

**经济学直觉**：OFI 捕捉*触点上可执行供需的净变化*，包含限价挂单与撤单里的信息——不只是
已成交量。买方 OFI 削薄卖档/堆积买档，机械地把价格往上推。这是供需压力机制而非基本面信息
机制，因此短 horizon 强、并会均值回归。

**证据**：Cont, Kukanov & Stoikov (2014, *J. Financial Econometrics*) 是标准参考：
线性 `ΔP = β·OFI` 在秒级 horizon 上 **R² 典型 0.4–0.7**，且 OFI 的解释力优于带符号成交量。
horizon 拉长后 R² 下降（信息/逆向选择效应盖过机械压力）。**这是 15s–60s horizons 的
第一优先级因子。**

**衰减/半衰期**：信号在 1–10s 最强，几十秒到一两分钟内基本消散；已实现流的*冲击*衰减更慢
（幂律，见 §6）。把 OFI 当作短 horizon（15s/30s）骨架，300s 起预期显著衰减。

**可行性**：快照近似；**逐笔流精确**。（本项目 `ofi_60s` 为逐笔实现。）

---

## 2. Queue / Depth Imbalance

### 2a. 最优档深度不平衡（queue imbalance, OIR）

**公式**：`OIR = (q^b - q^a) / (q^b + q^a)  ∈ [-1, +1]`

**L2 字段**：`BidVolume0`, `AskVolume0`（快照）或最优队列量（tick）。

**经济学直觉**：买队列厚而卖队列薄 → 吃掉卖档比砸穿买档省力 → 下一次价格变动更可能向上。
是 OFI 的潜在（未执行）对应物。

**证据**：Cont & de Larrard (2013, *SIAM J. Financial Mathematics*, arXiv:1104.4592) 在
两队列马尔可夫模型中证明**下一次 mid 变动的概率是队列不平衡 `I = q^b − q^a` 的函数**；
Cont, Stoikov & Talreja (2010, *Operations Research*) 给出底层随机订单簿动力学。
经验上 OIR 在 **~10s 到几分钟**内可预测，~10–15 分钟后衰减。

### 2b. 多档加权深度不平衡（WDI）

**公式**：`WDI = Σ_{k=1..K} w_k · (V^b_k − V^a_k)`，`w_k` 随 k 递减（如 `exp(−k/ℓ)`）。

**L2 字段**：`BidVolume0..9`, `AskVolume0..9`（快照直接给 10 档）。

**经济学直觉**：靠近价差的深度对紧邻的下一 tick 更重要；加权反映近簿流动性最先被消耗。
实务上前 3–5 档携带大部分信号，更深档更噪/受冰山单污染。

**证据**：Cao, Hansch & Wang (2009, *J. Financial Markets*) 证明 LOB 深度（及其不平衡）
在秒到几十分钟 horizon 上预测收益；Cartea 等（*Algorithmic and High-Frequency Trading*,
Cambridge, 2015）形式化加权不平衡并证明多档加权优于最优档。**衰减：秒到分钟。**

**可行性**：**完全来自快照**（10 档）。与 microprice 并列最干净的 snapshot-native 方向家族。

---

## 3. Trade / Order Imbalance

**公式**：`TI = (V_buy − V_sell) / (V_buy + V_sell)`，其中 `V_buy`, `V_sell` 为窗口内
按主动方分类的买/卖量。委托计数版本用新增买/卖委托数。

**L2 字段**：逐笔流 + 主动方 —— `TrdBSFlag` (B/S) 与 `Volume` 给*精确*分类（优于
Lee-Ready）。仅快照时对 `LastPrice` 用 tick/quote rule（粗）。

**经济学直觉**：净主动买入消耗卖档、抬升价格；不平衡是直接的压力/信息代理。quote/depth
不平衡在*最短* horizon 占优（执行前的潜在流）；trade imbalance 反映已被部分吸收的流，
但在有信息时携带更多*永久*信息。

**证据**：Chordia, Roll & Subrahmanyam (2002, *J. Financial Economics*)：order imbalance
日内影响收益与流动性；预测含量短命且部分被成本吃掉。方向：极短 lag 延续，分钟–小时级反转。

**衰减/半衰期**：秒到低分钟级延续；反转分量在分钟级浮现。主要映射 15s–60s，**注意 300s/900s
附近可能变号**。**可行性**：逐笔流精确；快照粗。

---

## 4. Microprice / Imbalanced Price (Stoikov)

**公式**：`P_micro = (P^a · q^b + P^b · q^a) / (q^b + q^a)` — 注意**交叉加权**：买量大把
估计拉向卖价（上），卖量大拉向买价。

**L2 字段**：`BidPrice0`, `AskPrice0`, `BidVolume0`, `AskVolume0`（快照）—— 完全
snapshot-native。

**经济学直觉**：microprice 是深度加权的"公允下一成交价"，是下一次 mid 的无偏、低方差、
不平衡调整预测：簿单边时，真实的短期价格已经部分移向拥挤的一侧。

**证据**：Stoikov (2018, *Quantitative Finance*, "The micro-price...")：microprice 在
亚秒到秒级 horizon **优于 mid 作为下一价格预测**，且增量具方向对称性。`microprice − mid`
差本身就是归一化不平衡信号。在紧价差、one-tick-spread 标的（流动性好的 ETF 常见 regime）
最有效。

**衰减/半衰期**：亚秒到几秒 —— 衰减最快的有用信号；主要对应 15s 标签，以及作为慢
horizon 的快速特征。**可行性**：**完全来自快照。**

---

## 5. VPIN（Volume-Synchronized Probability of Informed Trading）

**公式**：`VPIN = Σ_{i=1..N} |V^B_i − V^S_i| / Σ_{i=1..N} (V^B_i + V^S_i)`，在滚动的
`N`（~50）个**等成交量 bucket** 上计算（成交可跨 bucket 切分）。买/卖划分用主动方标记
（tick 流）或 BVC/tick rule（快照）。用**绝对差之和**（非净值），单边流无论方向都推高 VPIN。

**L2 字段**：tick 流 + 主动方优先；快照 + 对 `LastPrice` 的 BVC 近似。

**经济学直觉**：VPIN 估计单边/知情流占比（毒性）。高 VPIN ⇒ 做市商面临逆向选择 ⇒ 扩价差、
撤深度 ⇒ 流动性坍缩、波动上升。

**证据**：Easley, López de Prado & O'Hara (2012, *RFS*)：VPIN 在 2010 Flash Crash 前上升，
预测波动/流动性崩溃。**重要告诫**：(a) 是**波动率/regime**指标，不是干净的秒级方向信号；
(b) 天然窗口（≈50 buckets ≈ 小时级）比我们的 horizons 长——瞄准分钟级要用短滚动窗口；
(c) Andersen & Bondarenko (2014) 认为 VPIN 的预测力主要由量/波动驱动，应当作条件/风险变量。

**衰减/半衰期**：regime 级。**可行性**：tick 流精确；快照 BVC 近似。

---

## 6. Kyle Lambda 与价格冲击

**公式 (Kyle, 1985)**：`Δp_t = λ · q_t + ε_t`，`q_t` = 带符号成交量，
`λ = Cov(Δp_t, q_t) / Var(q_t)`（滚动 OLS）。Hasbrouck (1991, *J. Finance*) 用 VAR 分离
永久（信息）与暂时冲击；propagator / 平方根冲击文献（Bouchaud, Gefen, Potters & Wyart 2004,
*Quantitative Finance*）把 meta-order 冲击建模为 `I(Q) ∝ σ·√(Q/V)`，松弛核 `G(t) ∝ t^{−1/2}`。

**L2 字段**：tick 流：带符号成交量（`TrdBSFlag`, `Volume`）与价格变化；快照用 Lee-Ready
符号近似。

**经济学直觉**：λ 是单位净订单流的边际价格冲击 —— 市场深度/知情交易强度的直接度量。高 λ ⇒
流动性薄/知情流多。主要是**流动性风险与缩放变量**（也用作流因子的归一化分母），不是裸方向
预测。

**证据**：λ 稳定为正且时变（日内 U 形）。滚动 λ 预测后续变动幅度与执行成本。估计告诫：
OLS λ 有内生性偏误（用 IV/2SLS 或 Hasbrouck VAR）；Lee-Ready 误分类会偏置 λ（用
`TrdBSFlag`）。

**衰减/半衰期**：冲击本身按幂律 `~t^{−1/2}` 在秒到分钟上衰减（慢、长记忆）；λ 的*水平*是
慢变 regime 变量。**可行性**：tick 流最佳；快照粗。

---

## 7. Amihud Illiquidity

**公式**：`ILLIQ_t = |R_t| / ($Volume_t)`，窗口内取均值 —— 每元成交的价格冲击。

**L2 字段**：收益与量 —— 快照（`LastPrice`, `TradeVolume`）或 tick（`TrdMoney`，更精确）
均可计算。

**经济学直觉**：衡量每单位成交额价格动多少 —— 纯流动性/冲击计。高非流动性 ⇒ 给定流下更大
变动 ⇒ 更大未来波动，横截面上有流动性溢价。

**证据**：Amihud (2002, *J. Financial Markets*) 确立 ILLIQ 为标准流动性代理并有定价溢价；
Acharya & Pedersen (2005)、Pástor & Stambaugh (2003) 扩展为流动性风险框架。在我们的
horizons 它是**波动/幅度预测与条件变量**，非方向信号。

**衰减/半衰期**：慢变 regime 变量（日内持续）。**可行性**：**完全来自快照**。（v1.1/v2 hook）

---

## 8. Bid–Ask Spread 分解

**Roll (1984) 有效价差**（仅需成交价）：`s_Roll = 2·sqrt(−Cov(Δp_t, Δp_{t-1}))`
（协方差为负时）。

**Glosten & Harris (1988)** 用带符号成交量回归把价差分解为**永久/逆向选择**与
**暂时/订单处理**成分：
`Δp_t = θ0·q_t + θ1·q_{t-1} + (λ0 + λ1·q_t)·v_t + (γ0 + γ1·q_t)·v_{t-1} + ε_t`，
其中 λ（量×符号）项捕捉逆向选择。**Huang & Stoll (1997, *RFS*)** 扩展为三分量
（逆向选择、库存持有 Ho & Stoll 1981、订单处理）；Madhavan, Richardson & Roomans (1997,
*RFS*) 给日内逐笔版本。

**L2 字段**：Roll 只需 `LastPrice`（快照可）；GH/Huang-Stoll 需逐笔带符号量 + 报价。

**经济学直觉**：价差中*逆向选择*份额反映知情交易风险，*暂时*份额反映库存/订单处理。逆向选择
份额上升 → 信息到达 → 预测持续性；暂时份额上升 → 预测均值回归。价差水平本身是流动性/波动
预测（价差宽 ⇒ 簿薄 ⇒ 变动大）。

**证据/衰减**：分解稳健但在我们 horizons 主要是**流动性/regime 描述**；次秒采样会退化
（延迟/隐藏单），在 1–5s 聚合上估计。价差水平日内持续。**可行性**：Roll 快照；全分解 tick 流。

---

## 9. Order-Book Slope & Shape

**公式（代表）**：slope = 每 tick 的深度；shape 用限价单放置密度的幂律：
`depth profile: V(distance)；placement density p(Δ) ∝ Δ^{−γ}, γ ≈ 1.5`。
实用特征：累计深度 vs 价格距离的 log-slope、曲率、level-k/level-1 深度比、局部 slope 相对
滚动常态的偏离。

**L2 字段**：`BidPrice[0..9]`, `BidVolume[0..9]`, `AskPrice[0..9]`, `AskVolume[0..9]`,
`TotalBidVolume/TotalAskVolume`（快照）。

**经济学直觉**：Bouchaud, Mézard & Potters (2002) 与 Zovko & Farmer (2002, "The power of
patience")：簿有特征性重尾深度轮廓。触点附近比常态*更陡* ⇒ 集中的支撑/阻力；*更平/更薄* ⇒
给定流下更高扩散（波动）。Zovko–Farmer：波动 ∝ 1/深度 —— 簿变薄先于波动放大。Shape 更是
**波动/冲击**预测而非方向预测，但局部偏离幂律 slope 可提示即将到来的流动性驱动变动。

**证据/衰减**：shape 在 session 内准平稳（做条件/归一化特征）；偏离在秒到分钟有信息量。因为
shape 是普适统计规律（γ≈1.5），特质信息少于 imbalance —— 用于缩放/风险而非裸方向。

**可行性**：**完全来自快照。**

---

## 10. Order Arrival & Cancel Rates（与 Resiliency）

**公式（代表）**：分边/分档到达率 λ_arr、撤单率 λ_cxl、cancel-to-trade 比；resiliency =
大单后深度恢复速度（最初 ~100–500ms 的回补量/撤单量）。

**L2 字段**：仅 tick 流 —— 需 `Trade2_Order1`（委托 vs 成交）、挂撤与 `OrdNo` 匹配。快照
只有*净*深度变化。

**经济学直觉**：委托到达 = 新意图；撤单 = 撤退或"嗅探"。单边高 cancel-to-trade 标记幽灵
流动性/spoofing（反向信号）。成交后 resiliency 区分知情与噪声：**快速回补 ⇒ 变动是暂时的
（均值回归）；慢/部分回补 ⇒ 成交有信息（延续）**（Foucault, Kadan & Kandel 2005, *RFS*;
Obizhaeva & Wang 2013, *J. Financial Markets*）。

**证据/衰减**：前 100–500ms 测得的 resiliency 预测未来 1–5 分钟收益是反转还是延续 —— 直接
关联 300s/900s horizons。撤单率尖峰是秒级预警。**可行性**：**仅 tick 流。**

**本项目现状**：SSE 的 cancel 解码不可靠，`order_arrival_60s`/`cancel_ratio_60s` 在 SSE
恒为 NaN（仅 SZSE 有效）；另建议与 spoofing 过滤联用（A 股 ETF 可见深度常被操纵）。

---

## 11. Realized Volatility 及变体

**RV**：`RV_t = Σ_i r_{t,i}^2`（日内 mid/last 对数收益）。
**Bipower variation (BV)**（Barndorff-Nielsen & Shephard 2004，jump-robust）：
`BV = μ_1^{−2} · Σ_i |r_i|·|r_{i+1}|, μ_1 = sqrt(2/π)`。
**Realized kernel**（BNS-Hansen-Lunde-Shephard 2008, *Econometrica*）：核对加权收益自协方差
（Parzen / Tukey-Hanning），对 microstructure noise（bid-ask bounce）稳健。
**Realized semi-variance**（BNS-Kinnebrock-Shephard 2010; Patton & Sheppard 2015）：
`RSV⁻ = Σ_i r_i^2 · 1{r_i < 0}`（上行 RSV⁺）—— 下行（"坏"）半方差对未来波动与下跌预测更强。

**L2 字段**：`LastPrice` 或 mid 的收益（快照即可；3s 采样已抑制大部分微观噪声，kernel 去除
其余）。

**经济学直觉**：波动聚集且强持续 → 短 horizon 过去 RV/BV/kernel 是近期未来波动幅度的最佳
预测；半方差补充风险方向信息。是**幅度/风险预测**而非方向 —— 但可用于门控/缩放方向因子，且
下行 RV 自带风险溢价含量。

**证据/衰减**：波动高持续（慢衰减、长记忆）—— 用多尺度窗口 15s/30s/60s/300s/900s。3s 采样
用 realized kernel 避免噪声偏误；有跳跃时用 BV。

**可行性**：**完全来自快照**（tick-time 变体可选）。本项目 `rv_60s`/`rv_300s` 为快照实现。

---

## 12. Trade-Duration 因子（ACD）

**公式（Engle & Russell 1998, *Econometrica*）**：duration `x_i = t_i − t_{i−1}`；条件期望
duration `ψ_i = E[x_i | F_{i−1}]`，log-ACD：
`log ψ_i = ω + Σ_j α_j·log x_{i−j} + Σ_j β_j·log ψ_{i−j}`。
简易可用特征：窗口均值/中位 duration、duration z-score、当前 duration vs 滚动均值、ACD 残差。

**L2 字段**：仅 tick 流（逐笔时间戳 `TransactTime`），3s 快照不可恢复。

**经济学直觉**：duration 像波动一样聚集。短 duration = 高活跃/流动性；长 duration = 非流动。
期望 duration 是反向流动性代理与信息到达（突发）代理。价格冲击 ACD 扩展中，期望 duration
越长 ⇒ 单笔冲击越大（与 Kyle/Glosten-Milgrom 一致）。

**证据/衰减**：duration 日内持续（慢）；做流动性/活跃度条件变量与触发器（突发 = 短 duration
= 变动概率升高）。**可行性**：**仅 tick 流。**

---

## 13. Queue-Position / Queue-Reactive 因子

**概念**：queue-reactive 模型（Huang, Lehalle & Rosenbaum 2015, *J. Financial
Econometrics*）中，限价/市价/撤单到达强度依赖*当前*买卖队列量 `(q^b, q^a)`，给出马尔可夫
特征：挂单成交概率、触点移动概率、被动委托的 queue position 价值；Moallemi & Wang 扩展为
queue-reaction 信号。

**L2 字段**：仅 tick 流 —— 需要逐事件队列重建（挂单、撤单、成交、`OrdNo`、量）。

**经济学直觉**：把 §2 形式化为标定好的概率（而非启发式 OIR），是通往最优执行/做市模型的桥梁。

**证据/衰减**：亚秒到秒可预测（queue-reactive 信号快）；主要用于最短 horizons 与执行质量。
理论基础：Cont & de Larrard (arXiv:1104.4592)。**可行性**：**仅 tick 流。**

---

## 14. ETF 特有：IOPV Premium / Discount

**公式**：`Premium_t = (LastPrice_t − IOPV_t) / IOPV_t`（沪深交易所在快照流中发布 IOPV，
更新频率 ≤15s，详见 [02-etf-microstructure.md](./02-etf-microstructure.md)）。

**L2 字段**：`LastPrice`, `IOPV`（快照）。**Snapshot-native。**

**经济学直觉**：ETF 可偏离篮子实时公允价值；授权参与人的申赎套利把它拉回，但回拉非瞬时，且
IOPV 本身可能过期（成分股流动性差时尤甚）。premium 是均值回归的错误定价信号：宽 premium
预测下行压力（申购然后卖 ETF），反之亦然。也是 lead-lag 的条件变量（篮子/期货通常领先 ETF
几十到几百 ms）。

**证据**：ETF 错误定价部分可预测，高波动 regime 与低流动性篮子下更宽；多数持续小 premium 在
无套利带内 —— 使用需带成本阈值。直接关联 15s–900s horizons 与跨资产（指数/成分股）lead-lag
因子。**可行性**：**完全来自快照**（IOPV 在文件里）。注意 premium 卫生过滤
（停牌/涨跌停/QDII），见 02 号文档。

---

## 15. Horizons 与衰减总表

| 因子 | 最佳 horizon | 衰减/半衰期 | 方向性? | 快照 | 逐笔 |
|---|---|---|---|---|---|
| OFI (CKS) | 15s, 30s | 秒级；冲击尾部到分钟 | **是** | 近似 | **精确** |
| Microprice | 15s | 亚秒 → 秒 | 是 | **是** | 是 |
| 最优档 OIR | 15s–60s | 秒 → 几分钟 | 是 | **是** | 是 |
| 多档 WDI | 15s–60s | 秒 → 几分钟 | 是 | **是** | 是 |
| Trade imbalance | 15s–60s（300s 反转） | 秒 → 分钟 | 是 | 近似 | **精确** |
| Kyle λ | conditioning | 慢 regime | 否（缩放） | 近似 | 精确 |
| Amihud | conditioning | 慢 regime | 否（幅度） | **是** | 是 |
| Spread 分解 | conditioning | 持续 | 否（regime） | 近似 | 精确 |
| Book slope/shape | conditioning | 准平稳 | 否（波动） | **是** | 是 |
| Arrival/cancel/resiliency | 60s–900s | 秒–分钟 | 混合 | 否 | **是** |
| RV / BV / kernel / semi-var | 全部（幅度） | 慢/持续 | 否（风险） | **是** | 是 |
| Trade duration (ACD) | conditioning | 持续 | 否（流动性） | 否 | **是** |
| Queue position | 15s | 亚秒 → 秒 | 是 | 否 | **是** |
| VPIN | regime/vol | 长窗口 | 否（风险） | 近似 | 精确 |
| IOPV premium | 15s–900s | 均值回归 | **是** | **是** | n/a |

---

## 16. 实施优先级建议（调研结论）

1. **第一批 snapshot-native**：microprice−mid、最优档 OIR、多档 WDI、book slope、quoted
   spread、RV/kernel/semi-var、Amihud、IOPV premium —— 全部可直接从 `1_snapshot.csv.gz`
   计算（v1 已实现前 8 个中的相应部分）。
2. **第二批 full-tick**：精确 OFI、`TrdBSFlag` 精确 trade imbalance（优于 Lee-Ready）、
   order arrival/cancel、resiliency、ACD、queue position。
3. 方向因子当收益预测用；流动性/波动家族（Kyle λ、Amihud、spread 成分、book shape、RV、
   VPIN、duration）当条件/缩放与幅度/风险输入。
4. horizon 映射要有意识：OFI/microprice/OIR → 15s–30s；盯住 trade imbalance 在 300s 的
   变号；resiliency 与 IOPV premium → 300s–900s。
5. 3s 采样用 realized kernel（而非裸 RV）避免噪声偏误；有跳跃用 BV；多尺度窗口对齐五个
   horizons。
6. Kyle λ 用 IV/2SLS 或 Hasbrouck VAR（勿用裸 OLS）；符号分类依赖交易所主动方标记。
7. 若实现 VPIN：短滚动 volume-bucket 窗口（非标准 ~50 buckets），且只当波动/regime 变量。
8. IOPV premium 带无套利带阈值，警惕 IOPV 过期；配合跨资产 lead-lag。
9. **所有量级（OFI R²、衰减/半衰期、A 股 LOB alpha ~1–5s 衰减）必须在本 SSE/SZSE 数据上
   重估** —— 文献数字只是方向性基准，A 股 ETF 动力学（T+0 类别、0.001 tick、散户噪声、
   spoofing）与美欧证据不同。
10. 深度不平衡因子配 cancel-to-trade/挂单寿命过滤，防 spoofing/幽灵流动性。
11. 数据约束：一切输出写 `/data/factor_lzt`；channel→instrument 映射逐日变化，join 永远按
    InstrumentID。

### Smoke 日初步观测（20250701 上午段，单日、仅供量级参考）

eval 阶段 IC 表：`oir` 0.21–0.25（t 8–16）、`microprice_dev` 0.21–0.24、`wdi` 0.19–0.23、
`ofi_60s` 0.054–0.070、`iopv_premium` 正确为负、`quoted_spread` ≈ 0；stage1 有 18/60
因子×horizon 组合通过。**单日 IC 偏高是选择偏差 + 样本极小的产物，不可作为因子结论；
以多日 walk-forward 为准。**

---

## 17. References

- Amihud, Y. (2002). Illiquidity and Stock Returns: Cross-Section and Time-Series Effects. *Journal of Financial Markets*, 5(1), 31–56.
- Andersen, T. G., & Bondarenko, O. (2014). VPIN and the Flash Crash. *Journal of Financial Markets*.（批评：VPIN ≈ 量/波动代理）
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and Bipower Variation with Stochastic Volatility and Jumps. *Journal of Financial Econometrics*, 2(2), 1–37.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing Realized Kernels to Measure the Ex Post Variation of Equity Returns in the Presence of Noise. *Econometrica*, 76(6), 1481–1536.
- Barndorff-Nielsen, O. E., Kinnebrock, S., & Shephard, N. (2010). Measuring Downside Risk — Realised Semivariance.
- Bouchaud, J.-P., Gefen, Y., Potters, M., & Wyart, M. (2004). Fluctuations and Response in Financial Markets. *Quantitative Finance*, 4(2), 176–190.（propagator, `G(t)∼t^{-1/2}`）
- Bouchaud, J.-P., Mézard, M., & Potters, M. (2002). Statistical Properties of Stock Order Books. *Quantitative Finance*, 2(4), 256–265.
- Cao, C., Hansch, O., & Wang, X. (2009). The Information Content of the Limit Order Book: Evidence from NYSE Specialist Trading. *Journal of Financial Markets*, 12(2), 327–352.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.
- Chordia, T., Roll, R., & Subrahmanyam, A. (2002). Order Imbalance, Liquidity, and Market Returns. *Journal of Financial Economics*, 65(1), 111–130.
- Cont, R., & de Larrard, A. (2013). Price Dynamics in a Markovian Limit Order Market. *SIAM Journal on Financial Mathematics*, 4(1), 2–25. arXiv:1104.4592.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. *Journal of Financial Econometrics*, 12(3), 471–513.
- Cont, R., Stoikov, S., & Talreja, R. (2010). A Stochastic Model for Order Book Dynamics. *Operations Research*, 58(1), 217–233.
- Easley, D., López de Prado, M., & O'Hara, M. (2012). Flow Toxicity and Liquidity in a High-Frequency World. *Review of Financial Studies*, 25(5), 1457–1493.
- Engle, R. F., & Russell, J. R. (1998). Autoregressive Conditional Duration: A New Model for Irregularly Spaced Transaction Data. *Econometrica*, 66(5), 1127–1162.
- Foucault, T., Kadan, O., & Kandel, E. (2005). Limit Order Book as a Market for Liquidity. *Review of Financial Studies*, 18(4).
- Glosten, L. R., & Harris, L. E. (1988). Estimating the Components of the Bid/Ask Spread. *Journal of Financial Economics*, 21(1), 123–142.
- Hasbrouck, J. (1991). Measuring the Information Content of Stock Trades. *Journal of Finance*, 46(1).
- Huang, R. D., & Stoll, H. R. (1997). The Components of the Bid-Ask Spread: A General Approach. *Review of Financial Studies*, 10(4), 995–1034.
- Huang, S., Lehalle, C.-A., & Rosenbaum, M. (2015). Simulating and Analyzing Order Book Data: The Queue-Reactive Model. *Journal of Financial Econometrics*, 13(1), 107–138.
- Kyle, A. S. (1985). Continuous Auctions and Insider Trading. *Econometrica*, 53(2), 239–263.
- Madhavan, A., Richardson, M., & Roomans, M. (1997). Why Do Security Prices Change? A Transaction-Level Analysis of NYSE Stocks. *Review of Financial Studies*, 10(4), 1035–1064.
- Obizhaeva, A., & Wang, J. (2013). Optimal Trading Strategy and Supply/Demand Dynamics. *Journal of Financial Markets*, 16(1), 1–32.
- Patton, A. J., & Sheppard, K. (2015). Good Volatility, Bad Volatility: What Jumps Can Tell Us About Volatility Dynamics. *Review of Economics and Statistics*, 97(3).
- Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market. *Journal of Finance*, 39(4), 1127–1139.
- Stoikov, S. (2018). The Micro-Price: A High-Frequency Predictor of Price with Directional Symmetry. *Quantitative Finance*, 18(12).
- Zovko, I., & Farmer, J. D. (2002). The Power of Patience: A Behavioural Regularity in Limit-Order Books. *Quantitative Finance*, 2(5), 387–392.

### 来源与置信度说明

- OFI、microprice、OIR、VPIN、Amihud、Roll、RV/BV/kernel、semi-variance、ACD 的公式为标准
  形式且已交叉核对，实现照抄即可。
- 量级数字（秒级 OFI R² ~0.4–0.7；A 股 LOB alpha ~1–5s 衰减）是文献方向性基准 ——
  **必须在本数据集重估**；衰减/半衰期随标的与流动性变化。
- 调研中出现的二手/不可验证条目已被剔除；只保留确认真实的文献。

## 更新日志

- 2026-08-04: 由 `docs/microstructure_factors.md` 全量合并汉化入库；追加 §16 smoke 观测与
  本项目现状注记。原文件保留为 legacy 镜像。
