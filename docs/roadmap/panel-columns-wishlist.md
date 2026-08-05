# 面板列扩展清单（Panel Column Wishlist）

> 更新日期：2026-08-04
> 状态：规划中。原则：**所有新列一次引擎改动、一遍数据重跑全部产出**（共享原始数据流，边际成本≈0），绝不逐因子重跑。
> 当前面板 36 列（见 docs/knowledge/00-overview.md），下列字段在原始数据中存在但被丢弃，或需少量引擎逻辑派生。

## A. 原始字段直接保留（零逻辑成本，下次物化必带）

| 列名 | 来源 | 解锁的机制族 |
|---|---|---|
| `bid_num_orders[k]` / `ask_num_orders[k]` (k=0..9) | 快照 BidNumOrders[k] / AskNumOrders[k] | 平均每单规模（qty/orders）、单数不均衡、大单/散单结构 |
| `total_bid_vol` / `total_ask_vol` | 快照 TotalBidVolume / TotalAskVolume | 全簿不均衡（比五档更宽的流动性视角） |
| `cum_trade_vol` | 快照 TradeVolume（累计） | 日内成交量节奏、VWAP 偏离、放量/缩量状态 |
| `open_px` / `high_px` / `low_px` / `pre_close` | 快照 | 价格在日内区间的位置、隔夜跳空语境、动量参照 |
| `iopv_velocity` | 快照 IOPV 差分/Δt | 溢价变化速度（现仅有水平 iopv_premium），套利压力方向 |

## B. 需引擎逻辑派生（中等成本，一次批量实现）

| 列名 | 逻辑 | 解锁的机制族 |
|---|---|---|
| `order_gap_ms` / `trade_gap_ms`（滚动均值/方差） | 相邻委托/成交流水间隔 | 事件强度、Hawkes 自激代理、"活跃度骤变" |
| `big_trade_flow_buy` / `big_trade_flow_sell` | TrdMoney 超阈值的成交按方向累计（滚动窗） | 大额资金流向（机构活动探测） |
| `trade_size_mean` / `trade_size_max` / `trade_size_cv` | 滚动窗内单笔成交规模分布 | 成交颗粒度结构（散户化 vs 机构化） |
| `cancel_rate` / `cancel_depth_dist` | 撤单流（**仅 SZSE 可靠**，OrdType='X'；SSE 无可靠撤单标记） | 试探性挂单、幌骗探测、队列撤空速度 |
| `order_lifetime_ms` / `fill_ratio` | OrdNo 关联委托与其后成交（TrdBuyNo/TrdSellNo、OrderTrdVolume） | 挂单存活时间、被动成交难度（执行价值因子） |
| `book_event_intensity` | 每秒订单簿变更次数（逐笔驱动） | 信息到达速率，与快照 3s 节奏解耦 |
| `auction_phase` / `sec_since_open` / `min_to_close` / `post_lunch` | 时段编码（09:15-09:25 开盘集合竞价、14:57 收盘集合竞价、午休恢复） | 日内季节性、信息释放期 regime |

## C. 跨资产 / 外部数据（多资产阶段，当前不做）

| 列名 | 说明 |
|---|---|
| 关联 ETF lead-lag 收益 | 如宽基 ETF 与行业 ETF 的领先滞后结构（需多资产面板） |
| 成份股订单簿 → 理论 IOPV | 用成份股逐笔/快照重建 ETF 实时理论净值，预测 IOPV 与折溢价修复（用户提议，需成份篮权重数据） |
| 股指期货（IF/IC/IM）引导 | 外部数据接入后评估 |

## 成本与执行约定

- 新增列 = 一次 C++ 引擎改动 + 对目标日期区间**重跑一遍**；A/B 组所有列合并进同一次重跑。
- 重跑范围按需：单标的试点期只重跑该标的所需 channel；多资产期全量重跑一次后落缓存（考虑同时落解析后的列式中间格式到 `/data/factor_lzt`，此后不再碰原始 gz）。
- 每次物化在 meta/sidecar 记录列清单与版本，skip-if-done 按列清单失效。

## 物化记录

### 2026-08-05 第一批（iter-002 候选解锁，588000 试点）

已实现为 **opt-in 因子列**（`--factors` 显式指定；未进 `kDefaultFactorNames`，
避免使已产出的全市场 run 的 skip-if-done sidecar 失效）：

| 列 | 对应本清单条目 | 说明 |
|---|---|---|
| `cum_trade_vol` | A 组 cum_trade_vol | 快照 TradeVolume 直通；日内递减 ⇒ NaN |
| `trade_gap_ms` | B 组 trade_gap_ms | 距上笔成交毫秒数（瞬时值） |
| `n_trades_60s` | B 组（成交计数，trade_arrival_burst 所需） | trailing 60s 成交笔数 |
| `avg_trade_size_60s` | B 组 trade_size_mean | trailing 60s 平均单笔规模 |
| `large_trade_share_60s` | B 组（大单结构） | 最大 ceil(n/10) 笔的成交量占比（自归一化阈值） |

范围决定：本批只物化 library/candidates.json 两个 needs_materialization 候选
（`large_trade_share_60s`、`trade_arrival_burst`）所需列；A 组其余（bid/ask_num_orders、
total_bid/ask_vol、OHLC）与 B 组其余（big_trade_flow、order_lifetime、book_event_intensity、
时段编码）**留待多资产阶段一次性批量物化**（届时默认注册表扩容 + 全量重跑）。
固定金额阈值的大单列（`big_trade_flow_buy/sell`，TrdMoney 阈值）与
signed 大单占比变体作为后续候选，不在本批。

补记（同日）：replay 发现 `trade_gap_ms` 负值（20250702：16,022 行，min -970ms）。
根因为 SSE 发布批次内 UpdateTime 相位差 + 引擎共享归并游标（非 ETF barrier 快照
把 tick 游标拉过本标的快照时刻；详见 01-microstructure-factors.md 更新日志
2026-08-05（二））。修复：负值截断为 0（决策时刻该成交已知 ⇒ "刚发生"），
回归测试 `test_trade_gap_skew_clamp`。该截断属于本列语义约定；未来若需保留
负值的变体应另立新列。

补记（同日，物化完成与成本实测）：

- 全链路已完成并验证：引擎重建（#136）→ 缓存 replay 全通道重跑 66 交易日
  （stage1 缓存 build + stage2 replay，`--factors` opt-in 五列）→
  stage3 convert 重写 66 个 42 列分区（含 5380537 的非交易日 skip 修复）→
  服务器 `/data/factor_lzt/scripts/verify_mat17.py` 全 PASS
  （66 分区、42 列、trade_gap_ms ≥ 0、large_trade_share ∈ [0,1]、
  cum_trade_vol 日内单调、仅 588000）。
- **成本实测**（修正此前误引的 "0.4s/天"；0.4s 是 #94 审计中标的级
  replay 写 side-file 的测量值，convert 不读 side-file，物化走全通道 replay）：
  - stage1 缓存 build（一次性）：wall 104.8 min，均值 353.7s/天 CPU；
  - stage2 全通道 replay（因子列重算 + 重写生产 CSV）：wall 31 min，
    均值 102.8s/天 CPU，I/O 主导（每天 ~0.5-1GB 缓存读 + ~180MB CSV 写），
    与因子列数量基本无关；
  - stage3 convert：秒级/天。
- 两候选（large_trade_share_level、trade_arrival_burst）已于 eval v2 下
  完成首测，均 rejected（详见 /data/factor_lzt/iterations/eval_v2_iter002.md
  与 library/candidates.json，6e849bb）。

### 2026-08-06 第二批（iter-003 宽表迭代解锁，#144，588000 试点）

同样以 **opt-in 因子列**实现（不进 `kDefaultFactorNames`；`WISHLIST_FACTORS`
扩至 22 列，注册表保留名同步扩容）。面板由 42 列 → **59 列**
（CSV 58 列 + channel；314,053 行 × 66 交易日 × 仅 588000）。

| 列 | 组别 | 说明 |
|---|---|---|
| `total_bid_vol` / `total_ask_vol` | A 组 | 全簿买卖量合计（比 depth_*5 更宽） |
| `bid_orders5` / `ask_orders5` | A 组（名义） | 五档委托笔数合计——**SSE 上结构性为 0**（见下） |
| `open_px` / `high_px` / `low_px` / `pre_close_px` | A 组 | 日内参考价直通；high/low 为行情流滚动极值，因果成立 |
| `iopv_velocity` | A 组 | IOPV 60s 变化率（bps/s） |
| `ofi_15s` / `ofi_30s` | B 组 | 与 ofi_60s 同式的短窗订单流不平衡 |
| `trade_imbalance_15s` / `trade_imbalance_30s` | B 组 | 短窗主动成交不平衡 ∈[-1,1] |
| `buy_vol_60s` / `sell_vol_60s` | B 组 | 方向归因成交量（≥0） |
| `large_trade_net_share_60s` | B 组 | 大单（最大 ~10% 笔）**带符号**净占比 ∈[-1,1] |
| `book_event_intensity_60s` | B 组 | 每秒订单簿事件数（逐笔驱动） |

执行：stage2 replay `hftaf factors --dates 20250701..20250930 --cache use
--overwrite --workers 4`（仅 ch5 有缓存，其余 5 通道报
"cannot open cache meta" 属预期）+ stage3 `hftaf convert --instruments 588000
--overwrite`。schema 驱动（CSV 头 → Float64），python 层零改动直通。

**成本实测**：stage2 wall ≈ 40 min（34 个因子列；第一批 17 列时 31 min，
I/O 主导、随列数缓增）；stage3 convert 合计 ≈ 1 min。

**三个坑（务必记住）**：

1. **周六测试数据混入**：缓存目录含 5 个周六（20250809/20250823/20250906/
   20250920/20250927）的 SSE 测试行情。`--dates` 按缓存目录展开而非交易日历，
   replay 把这 5 天也物化了（raw+parquet 一度 71 天）。已删除这 5 天的派生输出
   （均为 /data/factor_lzt 下派生物，未触碰原始数据），恢复 66 天与第一轮
   评估全域一致。**教训：物化后必须核对分区集 == 交易日集；`ls` 缓存目录
   不等于交易日历。**
2. **bid_orders5/ask_orders5 在 SSE 上恒为 0**：SSE 快照无逐档委托数字段、
   亦无逐笔委托事件，`book.cpp` 的 num_orders 只在委托事件 insert_level 时
   累加。列保留（SZSE 阶段可能有效）但当前无效；"平均每单规模/单数不均衡"
   类想法改用成交口径（avg_trade_size_60s、n_trades_60s）。
3. **`set -e` 脚本别串 stage2**：无缓存通道使 factors CLI 退出码非 0，
   属预期失败；批处理脚本应在 stage2 后显式判断而非 `set -e` 直落。

验证：`/data/factor_lzt/scripts/verify_mat59.py` 全 PASS（66 分区、59 列、
仅 588000、314,053 行；取值范围与 warm-up 空值全部正常；orders5 检查改为
"恒 0 为预期、非 0 才报错"）。
