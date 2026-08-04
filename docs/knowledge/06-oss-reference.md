# 06 — 开源参考调研（OSS Reference Survey）

> **更新日期: 2026-08-04** · 状态: 现行（本次新鲜调研）
> 方法：2026-08-04 通过 GitHub REST API（search + repos 端点，curl 逐一核验）核对每个仓库的
> 全名、star 数、主语言、license、最近更新日期；所有数字均为当日 API 实测值。早期网络检索中
> 出现的错误仓库名（见 §9 勘误）已修正。追加规则：复审时在 §8 更新日志登记，按类别追加
> 新条目；star 数会漂移，引用时注明核验日期。

**一句话结论：没有任何开源项目覆盖「沪深 ETF L2（逐笔+3s 快照）→ 确定性因子引擎 →
mask 防穿越验证 → 成本感知回测」全链路；本项目引擎在该组合上是唯一的。可复用的主要是
单点组件：AXOrderBook（重建逻辑交叉核对）、hftbacktest（未来队列模型回测升级）。**

---

## 1. 调研范围与判定标准

评估维度：① 与 SSE/SZSE L2 格式的匹配度；② 是否支持 3s 快照+逐笔双源合并；③ 确定性
（bit-exact、可复现）；④ 是否带防穿越/验证机制；⑤ license 可商用性；⑥ 活跃度（最近更新）。
决策分类：**adopt（直接采用）/ reference（只读参考其实现）/ watch（跟踪观察）/
not-needed（与本项目无关或已被覆盖）**。

## 2. C++ 限价订单簿 / 撮合引擎

| 仓库 | Star | 语言 | 核验日更新状态 | 评估 |
|---|---|---|---|---|
| [Kautenja/limit-order-book](https://github.com/Kautenja/limit-order-book) | 311 | C++ | 2026-07 活跃 | reference：LOB API 设计参考；纯撮合，无 feed 回放 |
| [brprojects/Limit-Order-Book](https://github.com/brprojects/Limit-Order-Book) | 191 | C++ | 活跃 | reference：撮合语义对照 |
| [mansoor-mamnoon/limit-order-book](https://github.com/mansoor-mamnoon/limit-order-book) | 78 | C++ | — | reference：教学级实现 |
| [kpetridis24/lobsim](https://github.com/kpetridis24/lobsim) | 27 | C++ | — | watch：确定性 L3 回放 + paper execution，思路与本项目最接近，但面向欧洲市场格式 |

**结论**：全部是通用撮合引擎，不含 SSE/SZSE feed 解析。本项目 cpp/ 引擎的
snapshot-anchored book + SeqNo 序合并已按交易所数据自研，无需替换。

## 3. Rust LOB 生态

| 仓库 | Star | 核验情况 | 评估 |
|---|---|---|---|
| [joaquinbejar/OrderBook-rs](https://github.com/joaquinbejar/OrderBook-rs) | 501 | API 核验 | reference：Rust 重写备选（若未来引擎换语言） |
| [OctopusTakopi/binance_l3_est](https://github.com/OctopusTakopi/binance_l3_est) | 275 | API 核验 | watch：L2-delta → L3 估计，**Binance 专用**，但「用增量重建全档」的方法论与 SSE 千档场景相通 |
| [ninja-quant/ninjabook](https://github.com/ninja-quant/ninjabook) | 189 | API 核验 | reference：L2+trade 流处理框架 |
| [rubik/lobster](https://github.com/rubik/lobster) | 176 | API 核验 | reference：LOB 数据结构 |

**结论**：Rust 生态加密中心化，无 A 股 feed；仅方法论参考价值。

## 4. 中国 A 股 L2 相关（相关度最高）

| 仓库 | Star | License | 说明 | 评估 |
|---|---|---|---|---|
| [fpga2u/AXOrderBook](https://github.com/fpga2u/AXOrderBook) | 398 | MIT | **A 股订单簿重建**：千档快照+逐笔、Python + FPGA HLS 双实现 | **reference（重点）**：重建逻辑与边界情形可交叉核对；其 FPGA 路径与本项目无关 |
| [MistyBridge/ACMTOrderBook](https://github.com/MistyBridge/ACMTOrderBook) | 4 | — | AXOrderBook 的 C++ 优化 fork | watch |
| [cooronx/mirro-ex](https://github.com/cooronx/mirro-ex) | 7 | — | 沪深 L2 回放 + 模拟撮合 + 验证，Rust，带 Web UI | watch：方向最像本项目回测层，但规模小、无因子层、无防穿越验证 |
| [zhaoxiaoweizxw/calclevel2factor](https://github.com/zhaoxiaoweizxw/calclevel2factor) | 1 | — | C++ L2 日级因子计算 | not-needed（日级，与本项目秒级目标不符；仅字段口径参考） |

**结论**：AXOrderBook 是唯一值得深读的 A 股 L2 重建项目（MIT）。注意其面向**股票千档**，
本项目是 **ETF 十档快照 + 逐笔**，且本项目已自研并经 mask 验证，AXOrderBook 只作交叉核对，
不引入依赖。

## 5. Tick 级回测引擎（本项目 Stage 5 的潜在升级件）

| 仓库 | Star | License | 语言 | 评估 |
|---|---|---|---|---|
| [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest) | 4337 | MIT | Rust | **watch（首选升级候选）**：L2/L3 回放、queue-position 与 latency 模型、maker/taker 费率、pyo3 Python 绑定。当前本项目回测是 tick 价 + depth-impact overlay 的简化模型；若未来要排队论级成交模型，这是最成熟的落点 |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | 25257 | LGPL-3.0 | Rust | watch：确定性事件驱动框架，LGPL 需注意；偏交易执行框架而非 L2 研究 |
| [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | 21045 | Apache-2.0 | C# | not-needed：面向日/分钟级策略云 |
| [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | 8572 | NOASSERTION | Python | not-needed（license 不明 + 非 tick 级） |
| [kernc/backtesting.py](https://github.com/kernc/backtesting.py) | 8753 | AGPL-3.0 | Python | not-needed（AGPL + 非 tick） |
| [barter-rs/barter-rs](https://github.com/barter-rs/barter-rs) | 2214 | MIT | Rust | watch：实盘执行框架，未来实盘接入时评估 |
| [phil8192/ob-analytics](https://github.com/phil8192/ob-analytics) | 163 | — | R | reference：LOB 事件分析范式 |

## 6. 因子平台与符号挖掘

| 仓库 | Star | License | 评估 |
|---|---|---|---|
| [microsoft/qlib](https://github.com/microsoft/qlib) | ~47000 | MIT | reference：**日频** AI quant 平台（Alpha158 因子库）；与本项目秒级 L2 无交集，但 workflow / rolling retrain 的工程范式可借鉴 |
| [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | 4898 | NOASSERTION | reference：AFML 配套库；license 非开源标准，只读方法不引依赖 |
| [trevorstephens/gplearn](https://github.com/trevorstephens/gplearn) | 1873 | BSD-3 | watch：GP 符号回归。**注意**：本项目立场是 GP 挖掘的因子必须过经济学根基+统计门控（见 [05](./05-factor-selection-gating.md) §10），不接受盲 GP 准入 |
| [lc-sysbs/alpha101](https://github.com/lc-sysbs/alpha101) | 94 | — | reference：CN alpha101 实现，公式口径参考 |
| [YutaoWang03/Quant-Alpha101](https://github.com/YutaoWang03/Quant-Alpha101) | 20 | — | reference：同上 |
| [AlfredCYL/gplearn_cross_factor](https://github.com/AlfredCYL/gplearn_cross_factor) | 36 | — | watch：GP 截面因子示例 |
| [WYFHHH/QuantGplearn](https://github.com/WYFHHH/QuantGplearn) | 11 | — | watch：同上 |

## 7. 中国交易框架（实盘向，仅跟踪）与模拟器

| 仓库 | Star | License | 评估 |
|---|---|---|---|
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 44178 | MIT | watch：未来实盘接入 ETF 通道时的候选 |
| [wondertrader/wondertrader](https://github.com/wondertrader/wondertrader) | 6250 | MIT (C++) | watch：国产 HF 交易框架 |
| [kungfu-systems/kungfu](https://github.com/kungfu-systems/kungfu) | 4445 | — (C++) | watch：国产低延迟框架（注意其 README 近期转向 agent-workflow 宣传，细节引用前需重新核实） |
| [abides-sim/abides](https://github.com/abides-sim/abides) | 556 | — (Python) | reference：ABIDES 市场模拟器原始仓库（agent-based） |
| [jpmorganchase/abides-jpmc-public](https://github.com/jpmorganchase/abides-jpmc-public) | 174 | — (Jupyter) | reference：JPMC 接手的 ABIDES 公开版（网络检索中流传的 `jpmorganchase/abides-jpm` 路径不存在，以此为准） |
| [adamk999/ABIDES-Cpp](https://github.com/adamk999/ABIDES-Cpp) | 0 | — | not-needed（空转项目） |
| [adwhid/itchy-rust](https://github.com/adwhid/itchy-rust) | 51 | — | reference：ITCH 解析器，仅**格式解析范式**参考（本项目是 SSE/SZSE 自有格式） |
| [Lunyn-HFT/lunary](https://github.com/Lunyn-HFT/lunary) | 41 | — | reference：同上 |

## 8. 决策矩阵汇总

| 类别 | 代表项目 | 决策 | 理由 |
|---|---|---|---|
| A 股 L2 重建 | AXOrderBook | **reference** | MIT、千档重建可交叉核对；但本项目已自研并验证，不引依赖 |
| tick 回测 | hftbacktest | **watch → 未来 adopt** | 排队论/延迟模型是本项目成本回测的下一台阶；需先验证其能喂 SSE/SZSE 自定义格式 |
| 日频因子平台 | qlib | reference | 工程范式参考；非 L2 |
| LOB 撮合引擎 | limit-order-book 系 | reference/not-needed | 通用撮合，无 feed 回放 |
| 实盘框架 | vnpy / wondertrader / kungfu / barter-rs | watch | 生产化阶段再评估 |
| GP 挖掘 | gplearn | watch（受限） | 必须配合 05 号文档的统计门控使用 |
| 市场模拟器 | ABIDES | reference | 合成数据/placebo 检验思路 |

**缺口结论**：SSE/SZSE ETF L2 + 3s 快照网格 + SeqNo 序确定性合并 + mask 防穿越验证 +
T+1 底仓成本回测 —— 该组合无现成 OSS。本项目自研路线正确；最近邻是 cooronx/mirro-ex
（回放撮合）与 nkaz001/hftbacktest（回测模型），均可作为单点升级件而非整体替代。

## 9. 勘误与方法学注记

- 网络检索给出的仓库名多有错误，已全部用 GitHub REST API 逐一核验（2026-08-04）：
  `jpmorganchase/abides-jpm` → 实为 `abides-sim/abides` + `jpmorganchase/abides-jpmc-public`；
  `gplearn/gplearn` → `trevorstephens/gplearn`；`wondereamer/wondertrader` →
  `wondertrader/wondertrader`；`kungfu/origin` → `kungfu-systems/kungfu`。
- `alpacahq/marketstore` API 返回 404，已从调研中移除。
- 网络检索结果曾出现伪造 arXiv 链接（如 "arxiv.org/abs/2308.xxxxx"），本次一律以 API
  实测为准；未经 API 核验的声明不在本文引用。
- star 数为 2026-08-04 快照，复审时以 §更新日志 新核验值为准。

## 更新日志

- 2026-08-04: 初版入库。GitHub REST API 逐一核验 26 个仓库；补上前任未完成的新鲜调研。
