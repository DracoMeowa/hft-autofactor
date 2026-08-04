# hft-autofactor

> **更新日期: 2026-08-04**
> 中国 A 股 ETF 的自动化高频因子挖掘流水线：沪深 L2 原始行情（逐笔 + 3s 快照）→
> 确定性 C++ 因子引擎 → 防穿越 mask 验证 → parquet 面板 → 统计门控评估 → 成本感知回测。
> 项目地图见 [`docs/knowledge/00-overview.md`](docs/knowledge/00-overview.md)。

---

## ⛔ 绝对规则（不可协商）

1. **`/data/sse` 与 `/data/szse`（原始 L2 交易所 dump）永远只读**：不写入、不删除、
   不移动、不改内容。只读扫描可以。**一切输出只写 `/data/factor_lzt` 或本仓库。**
2. 不碰 `~/hft_etf`、`~/hft_etf-exp`、`/data/data_lzt`。
3. 所有 commit 以 **DracoMeowa** 身份（仓库本地 git 身份已配置）。
4. 共享服务器：`max_workers ≤ 4`，**不使用 GPU**。
5. 文档只**追加**不重写：`docs/knowledge/` 各篇末尾有更新日志，新内容追加到对应章节。

## 架构总览

```
 ┌────────────────────────── Stage 0 · 配置 ──────────────────────────┐
 │ config/pipeline.yaml: data_roots(RO) / out_root / engine_bin /     │
 │ horizons [15,30,60,300,900]s / factors []=全量12因子 / workers≤4   │
 └────────────────────────────┬───────────────────────────────────────┘
                              ▼
 ┌──────────── Stage 1 · hftaf-engine（C++17, cpp/）─────────────────┐
 │ 流式读 gz 逐笔+快照 → SeqNo 序合并 → snapshot-anchored 订单簿 →   │
 │ 12 因子注册表 + future-only 标签（ABSENT 语义，不前填）→          │
 │ 37 列 CSV + meta.json，原子 rename；flags: bit0 BOOK_UNSYNCED /   │
 │ bit1 SEQ_GAP_BEFORE / bit2 IOPV_INVALID / bit3 ONE_SIDED_BOOK     │
 └────────────────────────────┬───────────────────────────────────────┘
                              ▼
 ┌──── Stage 2 · mask 防穿越验证（validation/mask_test.py）──────────┐
 │ 按 SeqNo/时间位置截断重跑，前缀必须 bit 级一致；canary 因子必须   │
 │ 被抓出（证明检测功效）；trunc-PRESENT 必等于 full，trunc-ABSENT    │
 │ 允许（稀疏标的标签晚解析）                                         │
 └────────────────────────────┬───────────────────────────────────────┘
                              ▼
 ┌──────── Stage 3 · convert（ingest.py）────────────────────────────┐
 │ raw CSV → 日级 parquet 分区（dt=YYYYMMDD，37 列 schema）           │
 └────────────────────────────┬───────────────────────────────────────┘
                              ▼
 ┌──── Stage 4 · eval 门控（eval/{ic,splits,gating}.py）─────────────┐
 │ Spearman RankIC + Newey-West n_eff + ICIR → t-hurdle max(3,√2lnN) │
 │ + BHY-FDR q≤0.10 + 置换噪声底 99.9% + Deflated Sharpe；           │
 │ append-only TrialLedger；day-blocked purged walk-forward，         │
 │ OOS retention ≥ 0.5                                                │
 └────────────────────────────┬───────────────────────────────────────┘
                              ▼
 ┌──── Stage 5 · 成本感知回测（backtest/）───────────────────────────┐
 │ 三档佣金情景（institutional / retail_negotiated / retail_default）│
 │ 全存活才过门；T+1 底仓 sellable(t)=holdings(t-1)；tick ¥0.001 成交│
 │ depth-impact overlay；signal_lag ≥ 1 强制；沪市无收盘集合竞价 vs   │
 │ 深市 14:57–15:00 的差异处理                                        │
 └────────────────────────────────────────────────────────────────────┘

 探索支线（explore/）：原型因子注册表 → runner → causality 测试 → screen → CLI
 摘要支线（digest/）：coverage / data_quality / ic_decay / correlations / taxonomy
```

## 仓库结构

```
├── CMakeLists.txt              # 顶层：C++17、Release、确定性编译开关、ZLIB
├── config/pipeline.yaml        # 流水线配置（服务器从仓库根目录运行）
├── cpp/                        # C++ 因子引擎 hftaf-engine
│   ├── include/hftaf/          # book/decode/engine/factors/io/labels/output/session/types
│   ├── src/                    # 实现 + main.cpp CLI 驱动
│   └── tests/                  # ctest：decode/book/labels/factors/determinism
├── python/
│   ├── pyproject.toml          # deps: numpy/polars/pyarrow/pyyaml；scripts: hftaf, hftaf-backtest, hftaf-explore, hftaf-digest
│   ├── hft_autofactor/
│   │   ├── pipeline/           # cli + orchestrator（factors/convert/eval/mask 编排）
│   │   ├── validation/         # mask_test + golden（防穿越验证）
│   │   ├── eval/               # ic / splits / gating（Stage 4 统计门控）
│   │   ├── backtest/           # costs / execution / signals / engine / metrics / cli（Stage 5）
│   │   ├── explore/            # 探索支线：registry / runner / causality / screen / cli
│   │   ├── digest/             # 摘要支线：coverage / ic_decay / correlations / taxonomy / report
│   │   ├── ingest.py           # CSV → parquet
│   │   ├── reference_factors.py# 参考实现（差分校验用，见 04 号文档 §测试C 警告）
│   │   └── config.py
│   └── tests/                  # pytest（220+ 用例）
└── docs/
    ├── knowledge/              # ★ 知识库（00-06，追加式维护）+ etf_backtest_params.yaml 权威副本
    └── microstructure_factors.md / etf_cost_market_structure.md /
        etf_backtest_params.yaml  # 旧版镜像，内容已并入 knowledge/（勿再单独编辑）
```

## 构建

### C++ 引擎

要求：CMake ≥ 3.16、支持 C++17 的编译器、ZLIB。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # 5 个测试套件
```

**确定性契约**：顶层 CMake 已固化 `-O2 -fno-fast-math -ffp-contract=off`
（GNU/Clang；MSVC 为 `/O2 /fp:precise`）。价格用 int64 milli-CNY 表示。
同一代码重新编译的输出必须 bit 级一致（`test_determinism` 守护；跨构建 golden hash 已验证）。
**不要改动这些编译开关。**

### Python 层

```bash
pip install -e "python[dev]"     # 服务器在 conda env "autofactor"（py3.11）内执行
pytest python                    # testpaths = python/tests
```

## CLI 用法

### `hftaf`（Stage 1–4 编排，配置 `--config` 默认 `config/pipeline.yaml`）

日期语法：`YYYYMMDD[,YYYYMMDD|A..B]`，可混排，如 `20250701,20250703..20250705`。

```bash
# Stage 1: 跑 C++ 因子引擎（按 exchange×channel×date 并行，max_workers 见配置）
hftaf factors  --dates 20250701,20250702 [--channels 1,2,3] [--workers 4] \
               [--overwrite] [--dry-run]

# Stage 3: raw CSV → parquet 日分区
hftaf convert  --dates 20250701..20250705 [--overwrite] [--instruments 588000]

# Stage 4: IC/RankIC 评估 + 门控报告
hftaf eval     --dates 20250701..20250731 [--factors oir,ofi_60s] [--horizons 15,60,300] \
               [--instruments 588000]

# Stage 2: 防穿越 mask 验证（canary 必须失败）
hftaf mask     --dates 20250701 [--k 4]      # k = 每 job 截断点数
```

`--instruments`（convert/eval）把面板限制为单标的（如 588000 试点）：
convert 会把过滤值写进分区 sidecar（`factors.parquet.meta.json`），skip-if-done
据此区分「过滤分区」与「全市场分区」；eval 在标的数 < 5 时自动跳过截面 IC
（单标的无截面），全部门控改用按日时间序列 RankIC（Newey-West t）。

退出码：0 成功；1 有失败 job；2 配置/参数错误。

### `hftaf-engine`（C++ 底层驱动，通常由 orchestrator 调用）

```bash
hftaf-engine --exchange sse --date 20250701 --channel 3 \
  --ticks     /data/sse/202507/csv_XXXX/1_channel_3.csv.gz \
  --snapshots /data/sse/202507/csv_XXXX/1_snapshot.csv.gz \
  --out       /data/factor_lzt/raw/20250701/sse_ch3.csv \
  [--factors ofi_60s,oir] [--horizons 15,30,60,300,900] [--canaries] [--build-id SHA]
```

⚠ canary 名（`future_mid_15s` / `future_trade_sign`）只能随 `--canaries` 构建：
引擎在 `make_registry` 处**硬性拒绝** `--factors` 里不带 `--canaries` 的 canary 名
（退出码 2），防止泄漏列混入生产形状输出
（见 [`docs/knowledge/04-lookahead-prevention.md`](docs/knowledge/04-lookahead-prevention.md)）。

### `hftaf-backtest`（Stage 5）

```bash
hftaf-backtest --config config/pipeline.yaml \
  --factor oir --horizon 60 \
  --dates 20250701..20250731 \
  [--scenarios institutional,retail_negotiated,retail_default] \
  [--instruments 510300,510500] [--inventory 510300:100000] \
  [--params-yaml docs/knowledge/etf_backtest_params.yaml] \
  [--conservative-microfees] [--entry-z 2.0] [--exit-z 0.5] [--direction 1|-1] \
  [--max-units 100000] [--signal-lag 1] [--z-window 100] \
  [--min-net-sharpe 0.5] [--min-days 20] [--out DIR]
```

费用参数自动定位顺序：`docs/knowledge/etf_backtest_params.yaml`（**权威副本**）→
`docs/etf_backtest_params.yaml`（旧镜像）→ 相对 config 的仓库根；或显式 `--params-yaml`。
报告输出：`report.json`、`summary.csv`、`per_day_{scenario}.csv`、`equity_{scenario}.csv`；
成本门要求**三档佣金情景全部存活**（net Sharpe ≥ 0.5 且 ≥ 20 交易日）。

### `hftaf-explore`（探索支线：分钟级原型，parquet 面板上计算，不碰原始数据）

```bash
hftaf-explore list   [--config config/pipeline.yaml]
hftaf-explore add    --spec prototypes/my_idea.py [--overwrite]
hftaf-explore run    --dates 20250701..20250731 [--protos my_idea] \
                     [--chunk-days 5] [--k 4] [--overwrite]
hftaf-explore screen --dates 20250701..20250731 [--protos my_idea] \
                     [--horizons 60,300] [--max-abs-corr 0.85] \
                     [--min-is-t 2.0] [--min-oos-t 2.0] [--min-retention 0.5] \
                     [--embargo-days 1] [--n-test-days 5]
```

`run` 带 panel-prefix 因果检验：任一截断点不一致即整体拒绝并删除其分区；
`screen` 做 RankIC/NW + 库内去重 + purged IS/OOS 预筛。退出码：0 成功；
1 有拒绝/运行失败；2 用法/配置错误。原型规格持久化在
`{out_root}/explore/prototypes/`（内置原型见 `explore/registry.py`）。

### `hftaf-digest`（评估后反馈摘要，驱动下一轮假设）

```bash
hftaf-digest --out-root /data/factor_lzt [--report-dir DIR] \
             [--dates 20250701..20250930] [--eval-report PATH] \
             [--max-rows 200000] [--corr-threshold 0.7] [--no-panel]
```

读取 eval 产物（IC 表 / 门控报告 / trial ledger）与 parquet 面板，输出
一份 JSON + 一份中文 markdown 洞察报告（IC 衰减、pass/fail 归因、相关簇、
覆盖缺口、数据质量）到 `{out_root}/reports/digest/`。对流水线输出只读。

## 服务器工作流（ssh ETF）

```bash
ssh ETF
cd ~/hft-autofactor && conda activate autofactor
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build
hftaf factors --config config/pipeline-prod.yaml --dates 202507..202605  # 生产配置仅存服务器
```

- 引擎二进制：`~/hft-autofactor/build/cpp/hftaf-engine`（config 的 `engine_bin`）。
- 输出根：`/data/factor_lzt`（raw / parquet / validation / backtest / reports）。
- `config/pipeline-prod.yaml` 与 chain 脚本为**服务器专用、不入库**（引用服务器本地路径）；
  与已提交的 `config/pipeline.yaml` 的差异属有意为之。
- SSE 数据月份目录 202507–202605；instrument→channel 映射**逐日变化**，由 orchestrator
  按日发现。共享服务器：`max_workers ≤ 4`，不用 GPU。

## 贡献规则

1. **只追加不重写**：`docs/knowledge/` 各篇按章节追加并在文末更新日志登记日期；
   阈值/结论变更须注明依据。
2. 提交身份 DracoMeowa；commit message 说明动机。
3. 不在 `/data/sse`、`/data/szse`、`~/hft_etf*`、`/data/data_lzt` 产生任何写入。
4. 共享资源：并行度 ≤ 4、不占 GPU；长任务用 nohup + 日志落 `/data/factor_lzt`。
5. 改引擎编译开关/输出 schema/mask 语义前，先读 `docs/knowledge/00` 与 `04`。
6. 因子注册表新增因子：先补经济学根基（见 01 号文档体例）再过 05 号文档的统计门控。

## 知识库导航

| 文档 | 内容 |
|---|---|
| [00-overview](docs/knowledge/00-overview.md) | 项目地图：阶段/数据/horizon/绝对规则/schema/flag 位 |
| [01-microstructure-factors](docs/knowledge/01-microstructure-factors.md) | 因子公式 + 经济学含义 + 衰减证据 |
| [02-etf-microstructure](docs/knowledge/02-etf-microstructure.md) | IOPV 频率、T+0/T+1 分类、lead-lag、溢价卫生 |
| [03-ashare-etf-costs](docs/knowledge/03-ashare-etf-costs.md) | 全费用参数表（fee_table_v1 全文内嵌） |
| [04-lookahead-prevention](docs/knowledge/04-lookahead-prevention.md) | purging/embargo、mask 测试 A–E、对齐陷阱 |
| [05-factor-selection-gating](docs/knowledge/05-factor-selection-gating.md) | 门控阈值、多重检验栈、ledger 治理 |
| [06-oss-reference](docs/knowledge/06-oss-reference.md) | 开源生态调研（2026-08-04 新鲜核验） |

## 已知边界（2026-08-04）

- `order_arrival_60s` / `cancel_ratio_60s` 在 SSE 上恒为 NaN（撤单解码不可靠，仅 SZSE 有效）；
  SSE 上有效因子为 10/12。
- 501xxx LOF / 508xxx REIT / 货币 ETF（511xxx）流动性差，生产跑数应考虑流动性过滤。
- 稀疏标的的 `fwd_*` 标签在「首个 U ≥ t+H 的快照」处解析，实际 horizon 可能 > H（已文档化）。
- `reference_factors.py` 差分校验尚未接入流水线，且列名/语义与引擎有出入（见 04 §测试C）。

## 更新日志

- 2026-08-04: 重写 README（架构图、构建/CLI/服务器工作流、贡献规则、知识库导航）；
  补齐 docs/knowledge/05、06；`etf_backtest_params.yaml` 权威副本落位 knowledge/。
