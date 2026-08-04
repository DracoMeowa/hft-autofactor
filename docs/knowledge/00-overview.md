# 00 — 项目总览与知识库导航

> **更新日期: 2026-08-04** · 状态: 现行
> 本文是 hft-autofactor 知识库的入口。新迭代请在对应章节末尾**追加**，不要重写既有结论；
> 需要推翻旧结论时，保留原文并标注 `[DEPRECATED 2026-xx-xx]` 及原因。

## 1. 项目定位

**hft-autofactor** 是面向中国 A 股 ETF 的自动化高频因子挖掘流水线：从交易所 Level-2 原始
转储（逐笔委托/成交 + 3 秒十档快照）出发，单遍流式重建 per-instrument 订单簿状态，计算
有经济学根基的微观结构因子与严格 future-only 的前瞻收益标签，经过防 look-ahead 验证、
多重检验校正的统计筛选，最终在三档佣金情景 + A 股费用栈 + T+1 底仓约束下做成本感知回测。

- 资产范围 (v1): 仅 ETF — 上交所 `50/51/52/56/58xxxx`，深交所 `15/16xxxx`（`is_etf_code`
  目前也放行 501xxx LOF 与 508xxx REITs，流动性差，生产跑数建议加流动性过滤）。
- 预测 horizons: 前瞻 **15 / 30 / 60 / 300 / 900 秒**（mid 与 last 两套收益标签）。
- 输出网格: 每标的 **3 秒快照网格**，一行一个观测。

## 2. 绝对规则（不可协商）

1. **`/data/sse` 与 `/data/szse` 只读**。原始 L2 交易所转储禁止任何写入、删除、移动、内容
   修改；只读扫描允许。所有产出（raw/parquet/validation/reports/backtest/logs）一律写
   `/data/factor_lzt` 或本仓库。
2. 不得触碰 `~/hft_etf`、`~/hft_etf-exp`、`/data/data_lzt`。
3. 共享服务器，并行度保持克制：**max_workers ≤ 4**；v1 不需要 GPU。
4. Git 提交一律以 **DracoMeowa** 身份（仓库本地 identity 已配置）。
5. 数据完整性：已对原始数据目录建立 integrity manifest 并 `chmod -R go-w`；任何跑数前后
   可用 manifest 复核文件数与时间戳。

## 3. Pipeline 阶段总览

```
Stage 0  config & job discovery   (py-eval: config/pipeline.yaml → DayJob 列表)
            │  按 (date, exchange, channel) 切 job；instrument→channel 每日重发现
            ▼
Stage 1  因子/标签引擎            (cpp-core: hftaf-engine，单进程流式，CPU-only)
            │  gz 双流归并(SeqNo 序 + tie 约定) → snapshot-anchored book
            │  → 12 因子 + fwd_mid/fwd_last 标签(ABSENT 语义)
            │  → raw/YYYYMMDD/{ex}_ch{N}.csv (+ .meta.json，tmp+rename 原子写)
            ▼
Stage 2  look-ahead mask 验证     (py-eval: hftaf mask)
            │  按 SeqNo 位置截断输入 → 重跑同一二进制 → 前缀 bit 级一致
            │  canary 泄漏因子必须失败（证明验证器有检出力）
            ▼
Stage 3  parquet 转换             (py-eval: hftaf convert)
            │  全天 channel CSV → parquet/dt=YYYYMMDD/factors.parquet（幂等）
            ▼
Stage 4  评估与统计门控           (py-eval: hftaf eval)
            │  Spearman RankIC + Newey-West n_eff t 值 + ICIR
            │  → stage1 多重检验筛(t-hurdle/BHY-FDR/置换噪声底/DSR, TrialLedger 强制记账)
            │  → day-blocked purged walk-forward → stage2 OOS retention 门
            ▼
Stage 5  成本感知回测             (py-backtest: hftaf-backtest，仅对 admitted 因子)
            │  causal z-score → hysteresis 仓位 → tick 取整成交 + 深度冲击
            │  → 费用栈(3 佣金情景) + T+1 卖出锁(底仓链) + 分交易所收盘处理
            │  → 净 PnL/Sharpe/换手/回撤；gate_on_costs 要求全情景存活
            ▼
         /data/factor_lzt/backtest/ 报告
```

各 stage 均支持 **skip-if-done 断点续跑**（输出存在且 meta sidecar 与输入匹配即跳过）。

## 4. 数据

### 4.1 输入（只读）

```
/data/sse/YYYYMM/csv_MMDD_HHMMSS/
    1_channel_N.csv.gz    # 逐笔：委托+成交流（按 channel 分片）
    1_snapshot.csv.gz     # 3 秒十档快照（含 IOPV）
/data/szse/YYYYMM/...     # 同构（当前生产以 SSE 为主）
```

- SSE 可读月份: 202507 – 202605（2025-07-01 起）。
- **instrument→channel 映射逐日变化**，Stage 0 每天重新发现；下游一律按
  `InstrumentID` join，绝不按 channel join。
- 已知数据怪癖（已在 decode 层修复，见 git log）：
  - dump 的时间字段**丢弃前导零**（`91400650` = 09:14:00.650），按右对齐位置解析；
  - 快照列名带括号：`BidPrice[0]`、`AskPrice[0]`；
  - 09:15–09:25 集合竞价回放会产生时间倒挂（约 -600s），属良性，竞价 tick 不进入输出；
    **但引擎假设 session 内时间单调**——若未来出现盘中倒挂会静默污染 60s 窗口因子，需监控。

### 4.2 输出（全部在 /data/factor_lzt 下）

| 子目录 | 内容 |
|---|---|
| `raw/YYYYMMDD/` | 每 (date,exchange,channel) 一个排序 CSV + `.meta.json` |
| `parquet/dt=YYYYMMDD/` | 日分区 parquet 面板（**37 列**，见下） |
| `validation/` | mask 报告、golden hashes |
| `reports/` | eval 报告、`trial_ledger.jsonl`（append-only） |
| `backtest/` | `{factor}_h{horizon}/report.json + summary.csv + per_day/equity` |
| `logs/` | 每 job 日志 |

**Parquet 面板 schema (37 列)**:
`date, exchange, instrument, ts_ms, snap_seq, flags, mid_px, last_px, bid1_px, ask1_px,
bid1_qty, ask1_qty, depth_bid5, depth_ask5` + 12 因子列 + `fwd_mid_ret_{15,30,60,300,900}s`
+ `fwd_last_ret_{15,30,60,300,900}s` + `channel`。

**flags 位定义**: bit0 `BOOK_UNSYNCED`（重建簿与快照失配，待下次快照 resync）、
bit1 `SEQ_GAP_BEFORE`（channel SeqNo 缺口）、bit2 `IOPV_INVALID`、bit3 `ONE_SIDED_BOOK`。
**被标记区间的样本在评估中剔除，绝不插值/前向填充。**

## 5. Horizons 与标签语义

- `fwd_mid_ret_{H}s` / `fwd_last_ret_{H}s` = 同一连续 session 内**第一个** `ts >= t+H`
  的快照价格相对 t 时刻的收益。
- **ABSENT 语义**：窗口跨越午休 11:30–13:00、收盘竞价（深市 14:57–15:00）、session 结尾，
  或 t 时刻价格无效 → 标签为空（NaN），绝不 padding/forward-fill；**session 最后 15 分钟
  不产生 900s 标签**。
- **label-drift 注意**：对快照稀疏的标的（如 501xxx LOF），首个 >= t+H 的快照可能显著晚于
  t+H，因此 fwd_* 不是精确的 H 秒收益（已文档化，mask 验证按 directional 语义容忍）。

## 6. 默认因子注册表（12 个）

| 家族 | 因子 | 备注 |
|---|---|---|
| snapshot | `quoted_spread_ticks`, `microprice_dev`, `oir`, `wdi`, `book_slope`, `iopv_premium`, `rv_60s`, `rv_300s` | 3s 快照可直接计算 |
| tick | `ofi_60s`, `trade_imbalance_60s`, `order_arrival_60s`, `cancel_ratio_60s` | 需逐笔流 |

SSE 上 cancel 解码不可靠 → `order_arrival_60s` / `cancel_ratio_60s` **在 SSE 恒为 NaN**
（引擎打警告；SSE 有效因子集 = 10 个）。公式、经济学与衰减证据见
[01-microstructure-factors.md](./01-microstructure-factors.md)。

## 7. 运行环境

| 位置 | 说明 |
|---|---|
| 本地 (Windows) | `D:/claude/Quant_works/hft-autofactor`，remote `git@github.com:DracoMeowa/hft-autofactor.git`，branch `main` |
| 服务器 | `ssh ETF`（zt_li@117.50.177.32，NOPASSWD sudo）；clone 在 `~/hft-autofactor`；conda env `autofactor`（py3.11 + polars/pandas/pyarrow/pytest，~/miniconda3）；引擎二进制 `~/hft-autofactor/build/cpp/hftaf-engine` |
| 配置 | `config/pipeline.yaml`（提交版）；服务器另有不提交的 `config/pipeline-smoke.yaml` 与生产 chain 脚本（注意配置漂移风险） |

构建与 CLI 用法见 [README.md](../../README.md)。

## 8. 知识库导航

| 文档 | 内容 | 来源 |
|---|---|---|
| [01-microstructure-factors.md](./01-microstructure-factors.md) | 微观结构因子：公式 + 经济学 + 衰减证据（合并自 docs/microstructure_factors.md） | 文献调研 (2026-08) |
| [02-etf-microstructure.md](./02-etf-microstructure.md) | ETF 特有结构：IOPV 频率、T+0/T+1、lead-lag、premium 卫生过滤 | 调研 + 一手规则文件 |
| [03-ashare-etf-costs.md](./03-ashare-etf-costs.md) | A 股 ETF 费用与市场结构参数全表（合并 md + yaml） | 一手来源核验 (fee_table_v1) |
| [04-lookahead-prevention.md](./04-lookahead-prevention.md) | purging/embargo、mask test 设计、对齐陷阱、label-drift | 调研 + 工程实践 |
| [05-factor-selection-gating.md](./05-factor-selection-gating.md) | 阈值、多重检验栈、ledger 治理、退役机制 | 调研 + 已实现 GateConfig |
| [06-oss-reference.md](./06-oss-reference.md) | 开源生态调研：C++/Rust LOB 重建、中国 L2 因子仓库、tick 回测引擎 | 本次新鲜调研 (2026-08-04, GitHub API 逐一核验) |

旧文档 `docs/microstructure_factors.md`、`docs/etf_cost_market_structure.md`、
`docs/etf_backtest_params.yaml` 内容已全部合并进本知识库并保留原文件作为 legacy 镜像。
`hftaf-backtest` 的参数 yaml 自动发现**优先** `docs/knowledge/etf_backtest_params.yaml`
（2026-08-04 起为**权威副本**，与 docs/ 下镜像逐字节一致），再回落 `docs/`。
**维护规则：改参数只改 knowledge/ 权威副本，并同步镜像，勿让两者漂移。**

## 9. 已验证的里程碑（供后续迭代参照）

- 2026-08-03/04：服务器 clean rebuild（`-O2 -fno-fast-math -ffp-contract=off`）与旧构建
  **bit 级一致**（同 sha256，与 golden hash 相符）；ctest 5/5、pytest 124/124 通过。
- Smoke（20250701 SSE ch6 上午段）：289,200 行 / 144 标的 / 48.2s；eval IC 表合理
  （oir/microprice_dev/wdi 在 5 个 horizons 全部通过 stage1，ofi_60s 通过 15/30/60s）；
  mask 验证 4/4 截断点 PASS 且 canary 按预期失败；backtest 三情景费用分化正常
  （单日结果为负属噪声，不是因子结论）。
- 独立对抗性复核：13/13 手工 spot-check 与原始数据一致；生产路径未发现 look-ahead。

## 更新日志

- 2026-08-04: 知识库初版（librarian 恢复重建）。来源：上一轮工作流 journal 恢复 +
  仓库现状核验。
- 2026-08-04: 补齐 05（因子选择与统计门控）与 06（开源生态调研，GitHub API 逐一核验）；
  `etf_backtest_params.yaml` 权威副本落位 `docs/knowledge/`；README 同步重写。
  前任 librarian 遗留的缺口全部闭合。
