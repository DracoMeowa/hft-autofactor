# 03 — A 股 ETF 交易成本与市场结构参数全表

> **更新日期: 2026-08-04** · 状态: 现行（参数版本 **fee_table_v1**）
> 本文合并自原 `docs/etf_cost_market_structure.md`（完整来源表）与
> `docs/etf_backtest_params.yaml`（引擎可消费参数），两份原文件保留为 legacy 镜像
> （`hftaf-backtest` 两个路径都能发现）。追加规则：费率更新必须**升版本号**（fee_table_v2…）
> 并在文末更新日志登记，旧版本保留 —— A 股交易费有下调史（如 2023 印花税减半、过户费下调），
> 哪个费用版本产生哪个回测结果必须可审计。

**范围**: SSE + SZSE 上市 ETF · **用途**: hft-autofactor 回测引擎参数化
**约定**: bp = 0.01% = 万分之一；费率除注明外均为**单边**；主源优先，带 ⚠ 的条目存在来源
不确定性（§7 验证清单）。

---

## 1. 费用栈（二级市场买卖）

| 费用项 | 单边费率 | 备注 | 来源 |
|---|---|---|---|
| 印花税 | **0** | 《印花税法》应税证券仅股票与存托凭证，基金份额不在列（股票现行卖出单边 0.05%，2023-08-28 起减半） | 印花税法；财政部公告2023年第39号 |
| 经手费（交易所） | **0.4bp** (0.004%) | 基金竞价交易双向收取；**货币ETF、债券ETF暂免**；大宗交易费率下浮50% | 上交所收费一览表（2026-01更新，官方抓取）；深交所同率 ⚠（官网WAF，两处二手源一致） |
| 证管费（监管费） | 基准 0 / 保守 0.2bp | 费率 0.002% 双向，但对**基金交易**的适用性来源冲突；基准取 0，敏感性取 0.2bp ⚠ | 券商规费口径，待交割单验证 |
| 过户费 | 基准 0 / 保守 0.1bp | A股 0.001% 双向；ETF 是否收取来源冲突（一说仅跨市场ETF）⚠ | CSDC 口径，待交割单验证 |
| 佣金（券商） | 情景见下 | 双边收取；最低 5 元/笔（免五可谈） | 券商费率调研 |

**佣金情景（单边）**：

| 情景 | 费率 | 最低收费 | 适用 |
|---|---|---|---|
| `retail_default` | 2.5bp | ¥5/笔 | 散户默认（万2.5） |
| `retail_negotiated` | 1.0–1.5bp | ¥5/笔 | 活跃散户/渠道开户 |
| `institutional` | 0.5bp | 0（免五） | 机构/量化协议佣金 |

货币ETF佣金多数券商为 0；ETF 佣金普遍可单独谈到低于股票佣金。

**单边总成本公式**：

```
cost_per_side   = max(commission_rate × notional, min_commission)
                + 0.4bp × notional × 1{非货币/债券ETF} + 证管费 + 过户费
round_trip_cost = cost_buy + cost_sell + 滑点/价差成本
```

**数量级**：机构情景 ≈ 0.9bp/边（保守 +0.3bp → 1.2bp/边），散户默认 ≈ 2.9bp/边 + ¥5 最低。
**15s–300s 短周期因子边际通常 <3bp，费用模型直接决定因子存活** —— 三档情景必须全跑，
因子须在全部情景下净存活（`gate_on_costs` 已实现该逻辑）。

## 2. T+0 / T+1 规则

| 类别 | 回转交易 | 依据 |
|---|---|---|
| 股票ETF（宽基/行业/主题） | **T+1**（当日买入不得当日卖出） | 上交所交易规则 3.1.4；深交所同 |
| 债券ETF | T+0 | 交易规则 3.1.5；ETF实施细则 第24条 |
| 货币ETF | T+0 | 交易规则 3.1.5 |
| 黄金/商品ETF | T+0 | 交易规则 3.1.5；实施细则 第23条 |
| 商品期货ETF | T+0 | 交易规则 3.1.5 |
| 跨境ETF（QDII等） | T+0 | 交易规则 3.1.5（2023版优先于2020细则旧表述） |

**股票ETF的份额使用规则**（上交所ETF实施细则2020修订 第22条，沪市/跨市场指数ETF）：
当日**申购**份额同日可卖出不得赎回；当日**买入**份额同日可赎回不得卖出；当日赎回所得证券
同日可卖出不得用于申购；当日买入证券同日可用于申购不得卖出。详细套利回路见
[02-etf-microstructure.md](./02-etf-microstructure.md) §3。

**回测含义**：股票ETF日内回转必须依赖**底仓**：`sellable_qty(t) = holdings(t-1)`。买入
信号用现金建仓的同时，只能卖出昨日及更早的持仓。备选"合成T+0"路径 = 持有成分股→申购ETF→
当日卖出，成本 ≈ 申赎佣金≤0.5% + 两腿冲击，仅适合库存构建而非高频换手。

## 3. 撮合机制参数

| 参数 | 值 | 依据 |
|---|---|---|
| 最小报价单位 tick | **¥0.001**（基金） | 上交所交易规则 3.3.11；深交所同 |
| 相对tick | ≈2.5–3bp（价格¥3.5–4时） | 派生：0.001/price |
| 申报单位 | 100份整数倍；卖出零股可一次卖出 | 交易规则 3.3.8 |
| 单笔申报上限 | 100万份 | 交易规则 3.3.9/3.3.10（沪深同） |
| 大宗交易门槛 | ≥200万份 或 ≥200万元 | 交易规则 3.6.1 |
| 涨跌幅限制 | **±10%**（前收盘价），四舍五入到tick | 交易规则 3.3.13 + 3.3.17；深交所基金同。跟踪创业板/科创板的ETF仍为10%（20%档仅适用于重仓20%股票的LOF，非ETF） |
| 开盘集合竞价 | 9:15–9:25；9:20–9:25不可撤单；9:25撮合 | 交易规则 3.3.1 |
| 连续竞价 | 9:30–11:30, 13:00–15:00 | 交易规则 2.4.2 |
| **收盘机制（重要差异）** | 上交所基金：**无收盘集合竞价**，连续至15:00，收盘价=最后1分钟成交金额加权平均（4.1.3）；深交所：**14:57–15:00收盘集合竞价**，不可撤单 | 交易规则 4.1.3 / 深交所交易规则 |

## 4. 价差与深度假设（分层默认值）

⚠ 以下为方向性基准，**必须用自有 /data/sse、/data/szse L2 快照逐标的×时段实证校准**后替换
（校准输出只写 /data/factor_lzt）。

| 层级 | 判据（日成交额） | 报价价差 | 最优档深度 | 备注 |
|---|---|---|---|---|
| 高流动（510300/510050类） | >¥10亿 | 常压1个tick（2–3bp），时均1–2bp | ≥¥1000万/档，20+做市商 | 跟踪IOPV偏差<5–10bp |
| 中等（159915类） | ¥1–10亿 | 3–8bp，压力下10–15bp | ¥百万级 | 波动驱动 |
| 低流动（小众/迷你ETF） | <¥1亿 | 20bp–1%+ | 万元级/空心盘 | 有效价差可达头部5–20倍，依赖做市商报价 |

做市商义务锚点（《上交所基金做市业务指引》，⚠ 数值为综述口径）：双边报价时长占比 ≥90%
连续交易时段；最大报价价差按产品约定（头部约≤1%，其他2–3%）。日内形态：U形，
9:30–9:45 与 14:45–15:00 价差走阔。

**回测冲击模型**：对超过最优档深度 ~10% 的订单叠加 depth-aware impact（liquid ≥¥10M、
mid ~¥1M、illiquid 脆弱/空心盘，kappa 缩放、封顶 50bp）—— 静态价差假设会把低流动 ETF
成本低估 5–20 倍。

## 5. 申购赎回费用

| 项目 | 值 | 来源 |
|---|---|---|
| 代办券商申赎佣金 | **≤0.5%**（全包，含交易所/登记结算机构费用） | 159915招募说明书（2026-02更新版） |
| 最小申赎单位 | 基金合同约定，申报须为整数倍；159915 = **100万份** | 招募说明书；实施细则 第16条 |
| 场外现金申购费率 | 0.05%（159915） | 招募说明书 |
| 场外赎回费率 | 0.15%（159915） | 招募说明书 |
| 申赎申报 | 不可撤单（实施细则 第18条） | 上交所ETF实施细则 |

对二级市场 HFT 回测：申赎仅在库存再平衡/合成T+0建仓时进入成本模型，**不进信号 P&L**。

## 6. 回测操作建议（调研结论）

1. 三档佣金情景全跑并按**全情景净存活**门控因子 —— 15–60s horizons 的可行性在机构与
   散户档之间翻转。
2. 固定费用基线：印花税 0、经手费 0.4bp/边（货币/债券 ETF 为 0）、证管费 0、过户费 0；
   在拿到一份真实券商交割单前，保留 +0.3bp/边 的保守敏感性（`--conservative-microfees`）。
3. 执行模拟器强制股票 ETF T+1：锁定当日买入；底仓 T+0 实现为
   `sellable_qty(t)=holdings(t-1)`。**绝不允许股票 ETF 裸当日买卖回转** —— 这是 A 股 ETF
   回测最常见的错误。
4. 分交易所处理收盘：SSE 基金连续至 15:00（收盘=最后1分钟VWAP）；SZSE 14:57–15:00
   不可撤单收盘竞价。最后 3 分钟排除或单独分桶；不得无条件合并两所尾盘特征。
5. 价差/深度经验校准优先于本表分层默认值；tick 离散性显式建模（成交价取整到 0.001，
   报价价差下限 1 tick）—— 对流动 ETF 这是主要有效成本，队列位置成为 maker 成交的关键。
6. ±10% 涨跌幅按 tick 取整（规则 3.3.17）；创业板/科创板跟踪 ETF 仍 10%。
7. 申赎不进信号 P&L：≤0.5% 全包 + 100 万份起 + 不可撤单 → 只作底仓构建工具。
8. 参数集版本化入库（fee_table_v1_2026-08）。

## 7. 待验证清单（合计影响 ≤0.3bp/边，不阻塞回测）

1. 深交所官方收费页（szse.cn WAF拦截）：基金经手费 0.004% 及货币/债券 ETF 豁免 —
   已有两处独立二手源一致。
2. 证管费对基金交易的适用性（0 vs 0.2bp）：以一份真实券商 ETF 交割单定案。
3. 过户费对 ETF 二级交易的适用性（0 vs 0.1bp，或仅跨市场ETF）：同上。

## 8. 机器可读参数（fee_table_v1，引擎消费）

以下为 `etf_backtest_params.yaml`（fee_table_v1）的全量内容。文件本体有两份：
**权威副本 `docs/knowledge/etf_backtest_params.yaml`**（`backtest/cli.py::_find_params_yaml`
的优先路径）与 legacy 镜像 `docs/etf_backtest_params.yaml`（两者逐字节一致）。
`backtest/costs.py::load_cost_models` 与 `backtest/cli.py::_load_settlement_and_mechanics`
直接消费；修改时只改权威副本并同步镜像，升版本号：

```yaml
# fee_table_v1 — A-share ETF backtest engine parameters (SSE + SZSE)
# Sources & caveats: see docs/etf_cost_market_structure.md
# Rates are per-side unless noted. bp = 1e-4.
meta:
  version: fee_table_v1
  date: "2026-08-04"
  scope: ["SSE ETF", "SZSE ETF"]

fees:
  stamp_duty:
    rate_per_side: 0.0          # ETF units exempt (Stamp Duty Law: taxable = stocks/depository receipts only)
  handling_fee:                 # 经手费, exchange fee, per side
    rate_per_side: 0.00004      # 0.4bp, SSE official table (updated 2026-01); SZSE same rate (flag: secondary sources)
    exempt_categories: ["money_etf", "bond_etf"]   # SSE 暂免; SZSE consistent
    block_trade_rate_multiplier: 0.5
  regulatory_fee:               # 证管费
    rate_per_side_base: 0.0
    rate_per_side_conservative: 0.00002   # 0.2bp; applicability to fund trades unverified
    verify: true
  transfer_fee:                 # 过户费
    rate_per_side_base: 0.0
    rate_per_side_conservative: 0.00001   # 0.1bp; possibly only cross-market ETFs
    verify: true
  commission:
    scenarios:
      retail_default:    { rate_per_side: 0.00025, min_per_order_cny: 5.0 }
      retail_negotiated: { rate_per_side: 0.00010, min_per_order_cny: 5.0 }
      institutional:     { rate_per_side: 0.00005, min_per_order_cny: 0.0 }  # 免五 negotiable
    money_etf_commission: 0.0   # typical broker practice

cost_formula:
  per_side: "max(commission_rate * notional, min_commission) + handling_fee * notional * (1 if category not in [money_etf, bond_etf]) + regulatory_fee + transfer_fee"
  round_trip: "cost_buy + cost_sell + spread_or_slippage_cost"

trading_mechanics:
  tick_size_cny: 0.001
  lot_size_units: 100
  max_order_units: 1000000
  odd_lot: "sellable in one order"
  block_trade_min: { units: 2000000, notional_cny: 2000000 }
  price_limit:
    pct: 0.10                    # ±10% vs prior close, both exchanges, incl. ChiNext/STAR-tracking ETFs
    rounding: "round to tick (SSE rule 3.3.17)"
  sessions:
    opening_auction: { start: "09:15", end: "09:25", no_cancel_from: "09:20", match_at: "09:25" }
    continuous: { am: ["09:30", "11:30"], pm: ["13:00", "15:00"] }
    close:
      SSE: { closing_auction: false, close_price: "VWAP of last 1 minute (rule 4.1.3)" }
      SZSE: { closing_auction: ["14:57", "15:00"], no_cancel: true, close_price: "auction" }

settlement:
  t_plus_1: ["equity_etf"]       # bought units cannot be sold same day
  t_plus_0: ["bond_etf", "money_etf", "gold_etf", "commodity_etf", "commodity_futures_etf", "cross_border_etf"]
  equity_etf_share_rules:        # SSE ETF detail rules art.22 (Shanghai/cross-market index ETFs)
    subscribed_units: "sellable same day, not redeemable same day"
    bought_units: "redeemable same day, not sellable same day"
    redeemed_stocks: "sellable same day, not usable for subscription same day"
    bought_stocks: "usable for subscription same day, not sellable same day"
  backtest_inventory_rule: "sellable_qty(t) = holdings(t-1)"

creation_redemption:
  broker_fee_max: 0.005          # <=0.5% all-in incl. exchange/CSDC fees (159915 prospectus 2026-02)
  min_unit_example: { ticker: "159915", units: 1000000 }
  orders_cancellable: false
  otc_cash_subscription_fee: 0.0005   # 159915 off-exchange
  otc_cash_redemption_fee: 0.0015     # 159915 off-exchange

spread_depth_tiers:              # directional defaults — calibrate empirically from own L2 snapshots
  liquid:
    turnover_cny_day_gt: 1.0e+9
    quoted_spread: "1 tick (~2-3bp at 3.5-4 CNY), time-avg 1-2bp"
    depth_best_cny_ge: 1.0e+7
    example: "510300"
  mid:
    turnover_cny_day: [1.0e+8, 1.0e+9]
    quoted_spread_bps: [3, 8]
    stress_spread_bps: [10, 15]
    depth_best_cny: "~1e6"
    example: "159915"
  illiquid:
    turnover_cny_day_lt: 1.0e+8
    quoted_spread_bps: [20, 100]
    note: "hollow book, market-maker dependent; effective spread 5-20x liquid tier"
  intraday_pattern: "U-shaped; wider 09:30-09:45 and 14:45-15:00"

verification_flags:              # combined worst-case impact <=0.3bp/side; does not block backtesting
  - "SZSE official fee page inaccessible (WAF); 0.004% + bond/money ETF exemption corroborated by two independent secondary sources"
  - "regulatory_fee applicability to fund trades: settle with one real broker settlement statement"
  - "transfer_fee applicability to ETF secondary trades: settle with one real broker settlement statement"
```

## 9. 主要来源

- 上交所收费一览表（2026年1月更新）— sse.com.cn 官方页抓取
- 《上海证券交易所交易规则(2023年修订)》全文
- 《上海证券交易所ETF业务实施细则(2020年第二次修订)》全文
- 易方达创业板ETF(159915)招募说明书(2026-02-28更新) — static.cninfo.com.cn PDF
- 《中华人民共和国印花税法》应税税目表
- 深交所机制参数：两轮独立检索交叉验证（tick/手数/上限/涨跌幅/收盘集合竞价）

## 更新日志

- 2026-08-04: 初版入库（合并 `docs/etf_cost_market_structure.md` +
  `docs/etf_backtest_params.yaml`，fee_table_v1）。
- 2026-08-04: `etf_backtest_params.yaml` 权威副本落位 `docs/knowledge/`
  （与 `docs/` 镜像逐字节一致）；§8 说明同步更新。
