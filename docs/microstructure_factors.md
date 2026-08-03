# Market-Microstructure Factors for Short-Horizon ETF Return Prediction

**Project:** hft-autofactor — automated high-frequency factor mining for Chinese A-share ETFs
**Scope of this document:** Survey of economically-grounded microstructure factors computable from Level-2 tick and 3-second order-book snapshot data, with short-horizon (15s–15min) predictive relevance.
**Prediction horizons:** forward 15s, 30s, 60s, 300s (5min), 900s (15min).
**Asset universe (v1):** ETFs only (SSE 50/51/52/56/58xxxx, SZSE 15/16xxxx).

> Data notes: tick files `1_channel_N.csv.gz` carry per-order/per-trade rows (20 cols incl. `Trade2_Order1`, `Price`, `Volume`, `OrdSide`, `TrdBSFlag`, `TransactTime`); snapshot files `1_snapshot.csv.gz` carry 3-second 10-level books (~180 cols incl. `BidPrice0..9/BidVolume0..9/AskPrice0..9/AskVolume0..9`, `BidNumOrders0..9/AskNumOrders0..9`, `TotalBidVolume/TotalAskVolume`, `LastPrice`, `IOPV`). ETFs trade T+0 with 0.001 RMB tick.

---

## 0. Executive summary & feasibility map

The single most robust finding of the modern microstructure literature is that **contemporaneous order-flow imbalance (OFI) linearly drives mid-price changes** at second-scale horizons, with high R² that decays quickly as the horizon lengthens. Depth/queue imbalance is the "latent supply/demand" analogue and predicts the *next* price move. These two families — *realized flow* (trades/orders) and *latent flow* (book depth) — are the strongest directional signals at our horizons. Liquidity/volatility families (Amihud, Kyle lambda, spread components, realized vol) are best used as **conditioning/scaling variables and risk (variance) predictors**, not standalone directional predictors. Toxicity (VPIN) and duration/queue-position factors are regime/volatility indicators more than second-level directional alphas.

**Snapshot-only (3s book) vs full tick-stream feasibility:**

| Factor family | 3s snapshots only | Full tick stream |
|---|---|---|
| Depth / order-book imbalance (top-of-book & multi-level) | **Yes (primary)** | Yes |
| Order-book slope / shape | **Yes (primary)** | Yes |
| Microprice / imbalanced price | **Yes (primary)** | Yes |
| Bid-ask spread (quoted) | **Yes (primary)** | Yes |
| Quote-based realized vol / kernel / semi-variance | **Yes (primary)** | Yes (tick-time variant) |
| Amihud illiquidity | **Yes** (ret/volume bins) | Yes (finer) |
| ETF IOPV premium/discount | **Yes (primary)** | n/a |
| Trade / signed-volume imbalance | Approx. (Lee-Ready on `LastPrice`) | **Yes (exact, `TrdBSFlag`)** |
| Cont–Kukanov–Stoikov OFI | Approx. (net best-quote change; conflates trades & cancels) | **Yes (exact)** |
| Order arrival & cancel rates | No (net only) | **Yes** |
| Kyle lambda (regression) | Approx. | **Yes** |
| Spread decomposition (Roll / Glosten–Harris) | Approx. (Roll on `LastPrice`) | **Yes** |
| VPIN | Approx. (BVC on `LastPrice`) | **Yes** |
| Trade duration / ACD | **No** | **Yes** |
| Queue position / queue-reactive | **No** | **Yes** |
| Resiliency / replenishment after a large trade | No | **Yes** |

Rule of thumb: **depth/shape/price-level features come free from snapshots; anything needing event ordering, trade aggressor, cancellations, or durations needs the full tick stream.** The tick stream is strongly recommended for the flow- and queue-based factors; snapshots suffice to prototype the book-shape, microprice, volatility and ETF-premium factors.

---

## 1. Order Flow Imbalance (OFI) — Cont, Kukanov & Stoikov

**Formula.** OFI aggregates all best-quote events into net buying pressure. With best bid/ask price and size `(P^b, q^b)`, `(P^a, q^a)` at event/time `n`:

```
e^B_n = 1{P^b_n >= P^b_{n-1}} * q^b_n  -  1{P^b_n <= P^b_{n-1}} * q^b_{n-1}
e^A_n = 1{P^a_n <= P^a_{n-1}} * q^a_n  -  1{P^a_n >= P^a_{n-1}} * q^a_{n-1}
OFI_n = e^B_n - e^A_n
```

The core empirical relation is **linear price impact**:

```
ΔP_t = β · OFI_t + ε_t
```

**L2 fields needed:** Full tick stream — best bid/ask price & size on every order/trade event (from `1_channel_N.csv.gz`: `Price`, `Volume`, `OrdSide`, `Trade2_Order1`, `TransactTime`). Snapshots give only a *coarse* OFI (net best-quote size change between 3s frames), which conflates market orders with limit placements/cancels.

**Economic intuition.** OFI captures the *net change in executable supply/demand at the touch*, including the information in limit-order placements and cancellations — not just executed volume. A buy-side OFI thins the ask / builds the bid, mechanically pushing the price up. This is a supply-demand pressure mechanism, not a fundamental-information mechanism, which is why it is strong at short horizons and mean-reverts.

**Evidence.** Cont, Kukanov & Stoikov (2014, *Journal of Financial Econometrics*) is the canonical reference: linear `ΔP = β·OFI` fits mid-price changes with **R² typically ~0.4–0.7 at second-scale horizons**, and OFI dominates signed trade volume as an explanatory variable. R² falls as horizon grows because information/adverse-selection effects overtake mechanical pressure. This is the single highest-priority factor for the 15s–60s horizons.

**Decay / half-life.** Signal is strongest at 1–10s and largely dissipates within tens of seconds to a couple of minutes; the *impact* of executed flow decays slower (power-law, see §6). Treat OFI as the short-horizon (15s/30s) backbone and expect strong attenuation by 300s.

**Feasibility:** Approximate from snapshots; **exact from tick stream**.

---

## 2. Queue / Depth Imbalance

### 2a. Best-level depth imbalance (queue imbalance)

**Formula (order-book imbalance ratio, OIR):**

```
OIR = (q^b - q^a) / (q^b + q^a)      ∈ [-1, +1]
```

**L2 fields:** `BidVolume0`, `AskVolume0` (snapshots) or best-queue sizes (tick).

**Economic intuition.** A heavy bid queue vs a thin ask queue means less effort is needed to lift the ask than to hammer the bid → next price move more likely upward. It is the latent (not-yet-executed) analogue of OFI.

**Evidence.** Cont & de Larrard (2013, *SIAM J. Financial Mathematics*, arXiv:1104.4592) show in a two-queue Markovian model that the **probability of the next mid-price move is a function of queue imbalance** `I = q^b − q^a`; the direction is predictable while the queues are imbalanced. Empirically OIR predicts over **~10s to a few minutes**, decaying beyond ~10–15 min. Cont, Stoikov & Talreja (2010, *Operations Research*) establish the stochastic order-book dynamics underpinning this.

### 2b. Multi-level / weighted depth imbalance

**Formula:**

```
WDI = Σ_{k=1..K} w_k · (V^b_k − V^a_k) ,   w_k decreasing in k (e.g. exp(−k/ℓ))
```

**L2 fields:** `BidVolume0..9`, `AskVolume0..9` (snapshots give 10 levels directly).

**Economic intuition.** Depth near the spread matters more for the immediate next tick than depth far away; weighting reflects that near-book liquidity is what actually gets consumed first. Practitioners find top 3–5 levels carry most signal; deeper levels are noisier/iceberg-contaminated.

**Evidence.** Cao, Hansch & Wang (2009, *Journal of Financial Markets*) show limit-order-book depth (and its imbalance) predicts returns at horizons from seconds to tens of minutes. Cartea and co-workers (see *Algorithmic and High-Frequency Trading*, Cartea–Jaimungal–Penalva, Cambridge, 2015, and related papers) formalise weighted imbalance and show multi-level weighting beats top-of-book. **Decay:** seconds-to-minutes.

**Feasibility:** **Fully from snapshots** (10 levels). This is the cleanest snapshot-native directional family alongside microprice.

---

## 3. Trade / Order Imbalance

**Formula (trade imbalance):**

```
TI = (V_buy − V_sell) / (V_buy + V_sell)
```
where `V_buy`, `V_sell` are aggressor-classified buy/sell volume in a window. An order-count analogue uses new buy vs sell order counts.

**L2 fields:** Tick stream with aggressor side — `TrdBSFlag` (B/S) and `Volume` give *exact* classification (superior to Lee-Ready). From snapshots only, classify by tick/quote rule on `LastPrice` (coarse).

**Economic intuition.** Net aggressive buying consumes asks and lifts price; the imbalance is a direct pressure/information proxy. Quote/depth imbalance generally dominates at the *shortest* horizons (it is pre-execution latent flow); trade imbalance reflects flow already partly absorbed but carries more *permanent* information when informed.

**Evidence.** Chordia, Roll & Subrahmanyam (2002, *Journal of Financial Economics*) show order imbalance moves returns and liquidity intraday; predictive content is short-lived and partly consumed by costs. Direction: continuation at very short lags, mean-reversion over longer (minutes–hours) lags.

**Decay / half-life:** seconds to low minutes for continuation; reversal component emerges over minutes. Map primarily to 15s–60s, watch for sign-flip toward 300s/900s.

**Feasibility:** Exact from tick stream; coarse from snapshots.

---

## 4. Microprice / Imbalanced Price (Stoikov)

**Formula:**

```
P_micro = (P^a · q^b + P^b · q^a) / (q^b + q^a)
```
Note the **cross-weighting**: heavy bid volume pulls the estimate toward the ask (up), heavy ask volume toward the bid.

**L2 fields:** `BidPrice0`, `AskPrice0`, `BidVolume0`, `AskVolume0` (snapshots) — fully snapshot-native.

**Economic intuition.** The microprice is the depth-weighted "fair" next-trade price. It is an unbiased, lower-variance, imbalance-adjusted predictor of the next mid: when the book is one-sided, the true short-run price is already partway to the crowded side.

**Evidence.** Stoikov (2018, *Quantitative Finance*, "The micro-price: a high-frequency predictor of price with directional symmetry") shows the microprice **outperforms the mid as a predictor of the next price** over sub-second to second horizons, and its increment has directional symmetry. The `microprice − mid` gap is itself a normalized imbalance signal. Most effective in tight, one-tick-spread instruments (a common regime for liquid ETFs).

**Decay / half-life:** sub-second to a few seconds — the fastest-decaying useful signal; relevant mainly to the 15s label, and as a fast feature feeding slower horizons.

**Feasibility:** **Fully from snapshots.**

---

## 5. VPIN (Volume-Synchronized Probability of Informed Trading)

**Formula:**

```
VPIN = Σ_{i=1..N} |V^B_i − V^S_i| / Σ_{i=1..N} (V^B_i + V^S_i)
```
computed over a rolling window of `N` (~50) **volume buckets** (each bucket = fixed traded volume, trades split across bucket boundaries). Buy/sell assignment via aggressor flag (tick stream) or BVC / tick rule (snapshots). Uses **sum of absolute** differences (not net), so one-sided flow of either sign raises VPIN.

**L2 fields:** Tick stream + aggressor side preferred; snapshots + BVC on `LastPrice` approximate it.

**Economic intuition.** VPIN estimates the fraction of flow that is one-sided/informed (toxicity). High VPIN ⇒ market makers face adverse selection ⇒ they widen spreads and withdraw depth ⇒ liquidity collapses and volatility rises.

**Evidence.** Easley, López de Prado & O'Hara (2012, *Review of Financial Studies*, "Flow Toxicity and Liquidity in a High-Frequency World") show VPIN rose ahead of the 2010 Flash Crash and predicts volatility / liquidity breakdown. **Important caveats:** (a) it is a **volatility/regime** indicator, not a clean second-level *directional* signal; (b) its natural window (≈50 buckets ≈ hours) is longer than our horizons — use a short rolling window if targeting minutes; (c) Andersen & Bondarenko (2014) argue VPIN's predictive power is largely driven by volume/volatility, so treat it as a conditioning/risk variable rather than a standalone alpha.

**Decay / half-life:** regime-level; use as contemporaneous risk/volatility conditioning, not directional.

**Feasibility:** Exact from tick stream; approximate (BVC) from snapshots.

---

## 6. Kyle Lambda & Price Impact

**Formula (Kyle, 1985):**

```
Δp_t = λ · q_t + ε_t ,   q_t = signed volume
λ = Cov(Δp_t, q_t) / Var(q_t)     (rolling OLS)
```
Hasbrouck (1991, *Journal of Finance*) generalises this with a VAR that separates permanent (information) from transitory impact. The propagator / square-root-impact literature (Bouchaud, Gefen, Potters & Wyart 2004, *Quantitative Finance*; Bouchaud et al. *Trades, Quotes and Prices*) models impact of a meta-order as `I(Q) ∝ σ·√(Q/V)` with a relaxation kernel `G(t) ∝ t^{−1/2}`.

**L2 fields:** Tick stream: signed volume (`TrdBSFlag`, `Volume`) and price changes. Approximate from snapshots via Lee-Ready sign on `LastPrice`.

**Economic intuition.** λ is the marginal price impact per unit of net order flow — a direct measure of market depth / informed-trading intensity. High λ ⇒ thin liquidity / more informed flow. It is mostly a **liquidity-risk and scaling variable** (and a denominator to normalize flow factors), not a raw directional predictor.

**Evidence.** λ is reliably positive and time-varying (U-shaped intraday). Rolling λ predicts the *magnitude* of subsequent moves and execution cost. Estimation caveats: OLS λ is endogeneity-biased; use IV/2SLS or Hasbrouck VAR, and be aware Lee-Ready misclassification biases λ (use `TrdBSFlag`).

**Decay / half-life:** impact itself decays as a power law `~t^{−1/2}` over seconds-to-minutes (slow, long-memory); the *level* of λ is a slowly-moving regime variable.

**Feasibility:** Best from tick stream; coarse from snapshots.

---

## 7. Amihud Illiquidity

**Formula:**

```
ILLIQ_t = |R_t| / ($Volume_t)      ; aggregate as mean over a window
```
i.e. price impact per dollar of volume.

**L2 fields:** Returns and volume — computable from snapshots (`LastPrice`, `TradeVolume`) or tick (`TrdMoney`). Dollar volume = price × shares (use `TrdMoney` for exactness).

**Economic intuition.** Measures how much price moves per unit of money traded — a pure liquidity/impact gauge. High illiquidity ⇒ larger moves for a given flow ⇒ larger future volatility and (in cross-section) a liquidity premium.

**Evidence.** Amihud (2002, *Journal of Financial Markets*) establishes ILLIQ as the canonical liquidity proxy with a priced cross-sectional premium. At our horizons it is a **volatility/magnitude predictor and conditioning variable**, not a directional signal. Extended by Acharya & Pedersen (2005) and Pástor & Stambaugh (2003) into liquidity-risk frameworks.

**Decay / half-life:** slowly-moving regime variable (persistent within-day); use as scaling/risk input.

**Feasibility:** **Fully from snapshots.**

---

## 8. Bid–Ask Spread Decomposition

**Roll (1984) effective spread** (from transaction prices only):

```
s_Roll = 2 · sqrt( −Cov(Δp_t, Δp_{t-1}) )     (when covariance < 0)
```

**Glosten & Harris (1988)** trade-indicator regression decomposes the spread into a **permanent/adverse-selection** component and a **transitory/order-processing** component via signed trade volume:
```
Δp_t = θ0·q_t + θ1·q_{t-1} + (λ0 + λ1·q_t)·v_t + (γ0 + γ1·q_t)·v_{t-1} + ε_t
```
where the λ (volume × sign) terms capture adverse selection. **Huang & Stoll (1997, *RFS*)** extend to three components — adverse selection, inventory holding (Ho & Stoll 1981), order processing — via a VAR with trade indicators and quote revisions. Madhavan, Richardson & Roomans (1997, *RFS*) provide the intraday transaction-level analogue.

**L2 fields:** Roll — `LastPrice` only (snapshot-OK). Glosten–Harris / Huang–Stoll — per-trade signed volume + quotes (tick stream preferred).

**Economic intuition.** The *adverse-selection* fraction of the spread reflects informed-trading risk; the *transitory* fraction reflects inventory/order-processing. A rising adverse-selection component signals incoming information → predicts persistence; a rising transitory component predicts mean-reversion. The spread level itself is a liquidity/volatility predictor (wide spread ⇒ thin book ⇒ larger moves).

**Evidence.** Decompositions are robust but are primarily **liquidity/regime descriptors** at our horizons; use the adverse-selection share and spread dynamics as conditioning features. Note: standard decomposition models can degrade at sub-second sampling (latency/hidden orders), so estimate on 1–5s aggregation.

**Decay / half-life:** spread level is persistent intraday; component shares move with the information-arrival regime.

**Feasibility:** Roll from snapshots; full decomposition from tick stream.

---

## 9. Order-Book Slope & Shape

**Formula (representative):** slope = depth-per-tick across levels; shape via power-law of limit-order placement density
```
depth profile:  V(distance) ;  placement density p(Δ) ∝ Δ^{−γ},  γ ≈ 1.5
```
Practical features: log-slope of cumulative depth vs price distance; curvature; depth ratio level-k / level-1; deviation of the local slope from its rolling norm.

**L2 fields:** `BidPrice0..9`, `BidVolume0..9`, `AskPrice0..9`, `AskVolume0..9`, `TotalBidVolume/TotalAskVolume` (snapshots).

**Economic intuition.** Bouchaud, Mézard & Potters (2002, *Quantitative Finance*) and Zovko & Farmer (2002, *Quantitative Finance*, "The power of patience") show books have a characteristic heavy-tailed depth profile. A book *steeper* than normal near the touch implies concentrated support/resistance; a *flatter/thinner* book implies higher diffusion (volatility) for a given flow. Zovko–Farmer: volatility ∝ 1/depth — a thinning book precedes volatility expansion. Shape is more a **volatility/impact** predictor than a directional one, but local deviations from the expected power-law slope can flag imminent liquidity-driven moves.

**Evidence / decay:** shape is quasi-stationary within a session (useful as a conditioning/normalization feature); deviations are informative over seconds-to-minutes. Because the shape is a universal statistical regularity (γ≈1.5), it carries less idiosyncratic information than imbalance — use it for scaling/risk, not raw direction.

**Feasibility:** **Fully from snapshots.**

---

## 10. Order Arrival & Cancel Rates (and Resiliency)

**Formula (representative):** per-side/per-level arrival rate λ_arr, cancellation rate λ_cxl, cancel-to-trade ratio; resiliency = speed of depth recovery after a large trade (replenishment volume / cancellation volume in the first ~100–500ms).

**L2 fields:** Tick stream only — needs `Trade2_Order1` (order vs trade), order adds/cancels and `OrdNo` to match arrivals vs cancels. Snapshots reveal only *net* depth change.

**Economic intuition.** Order arrivals signal fresh intent; cancellations signal withdrawal or "sniffing." A high cancel-to-trade ratio on one side flags phantom liquidity / spoofing (a contrarian signal). Post-trade resiliency distinguishes informed from noise trades: **fast replenishment ⇒ the move was temporary (mean-reversion); slow/partial refill ⇒ the trade was informed (continuation)** (Foucault, Kadan & Kandel 2005, *RFS*; Obizhaeva & Wang 2013, *Journal of Financial Markets*).

**Evidence / decay:** resiliency measured over the first 100–500ms predicts whether the next 1–5 minute return reverts or continues — directly relevant to 300s/900s horizons. Cancel-rate spikes are second-scale warning signals.

**Feasibility:** **Tick stream only.**

---

## 11. Realized Volatility & Variants

**Realized variance/volatility (RV):**
```
RV_t = Σ_i r_{t,i}^2        (intraday returns r_{t,i} from mid or last)
```
**Bipower variation (BV)** — Barndorff-Nielsen & Shephard (2004, *J. Financial Econometrics*) — jump-robust:
```
BV = μ_1^{−2} · Σ_i |r_i|·|r_{i+1}| ,   μ_1 = sqrt(2/π)
```
**Realized kernel** — Barndorff-Nielsen, Hansen, Lunde & Shephard (2008, *Econometrica*) — robust to microstructure noise (bid-ask bounce) by kernel-weighting return autocovariances (Parzen / Tukey-Hanning kernels). **Realized semi-variance** — Barndorff-Nielsen, Kinnebrock & Shephard (2010) and Patton & Sheppard (2015) — downside-only:
```
RSV⁻ = Σ_i r_i^2 · 1{r_i < 0}      (RSV⁺ for upside)
```
Downside ("bad") semi-variance is the stronger predictor of future volatility and downturns.

**L2 fields:** Returns from `LastPrice` or mid (snapshots suffice; 3s sampling already dampens raw microstructure noise, and kernels remove the rest). Volume optional.

**Economic intuition.** Volatility clusters and is strongly persistent, so short-horizon past RV/BV/kernel is the best predictor of near-future volatility magnitude. Semi-variance adds direction-of-risk information. These are **magnitude/risk predictors**, not directional — but (a) they gate/scale directional factors, and (b) downside RV has its own risk-premium content.

**Evidence / decay:** volatility is highly persistent (slow decay, long memory — use multi-scale windows 15s/30s/60s/300s/900s). At 3s sampling use realized kernel to avoid noise bias; BV when jumps are present.

**Feasibility:** **Fully from snapshots.** (Tick-time RV variant available from the tick stream.)

---

## 12. Trade-Duration Factors (ACD)

**Formula (Engle & Russell 1998, *Econometrica*):** durations `x_i = t_i − t_{i−1}`; conditional expected duration `ψ_i = E[x_i | F_{i−1}]` with
```
log ψ_i = ω + Σ_j α_j · log x_{i−j} + Σ_j β_j · log ψ_{i−j}   (log-ACD)
```
Simple usable features: mean/median duration in a window, duration z-score, current-duration vs rolling mean, and the ACD residual.

**L2 fields:** Tick stream only — needs per-trade timestamps (`TransactTime`). Not recoverable from 3s snapshots.

**Economic intuition.** Durations cluster like volatility. Short durations = high activity/liquidity; long durations = illiquidity. Expected duration is an inverse liquidity proxy and a proxy for information arrival (bursts). In price-impact ACD extensions, longer expected duration ⇒ larger per-trade impact (consistent with Kyle/Glosten-Milgrom).

**Evidence / decay:** duration is persistent intraday (slow); useful as a liquidity/activity conditioning variable and as a trigger (a sudden burst = short durations = elevated probability of a move).

**Feasibility:** **Tick stream only.**

---

## 13. Queue-Position / Queue-Reactive Factors

**Concept.** In queue-reactive models (Huang, Lehalle & Rosenbaum 2015, *Journal of Financial Econometrics*), the arrival intensities of limit/market/cancel orders depend on the *current* bid/ask queue sizes `(q^b, q^a)`. This yields Markovian features: probability a resting order fills, probability the touch moves, and the *queue position* value of a passive order. Moallemi & Wang extend this to queue-reaction signals.

**L2 fields:** Tick stream only — needs per-event queue reconstruction (order adds, cancels, trades, `OrdNo`, sizes). Not from snapshots.

**Economic intuition.** Queue imbalance and queue-reactive intensities formalize §2: the next price move probability is a function of current queue state. These give a principled, calibrated probability (rather than a heuristic OIR) and are the bridge to optimal-execution/market-making models.

**Evidence / decay:** predictive at sub-second to seconds (the queue-reactive signal is fast); primarily for the shortest horizons and for execution quality. Cont & de Larrard (arXiv:1104.4592) provide the theoretical grounding.

**Feasibility:** **Tick stream only.**

---

## 14. ETF-specific: IOPV Premium / Discount

**Formula:**
```
Premium_t = (LastPrice_t − IOPV_t) / IOPV_t
```
(SSE/SZSE disseminate IOPV in the snapshot feed; cadence is seconds-scale.)

**L2 fields:** `LastPrice`, `IOPV` (snapshots). **Snapshot-native.**

**Economic intuition.** The ETF can deviate from the real-time fair value of its basket; creation/redemption arbitrage by authorized participants pulls it back, but the pull-back is not instantaneous, and IOPV itself can be stale (especially for less-liquid constituents). The premium is a mean-reverting mispricing signal: a wide premium predicts downward pressure (creations sell the ETF) and vice versa. It is also a conditioning variable for lead-lag (the basket/futures often lead the ETF by tens-to-hundreds of ms).

**Evidence:** ETF mispricing is partially predictable, wider in high-volatility regimes and for illiquid baskets; most persistent small premiums are inside the no-arbitrage band, so use it with a cost threshold. Directly relevant to 15s–900s horizons and to cross-asset (index/constituent) lead-lag factors.

**Feasibility:** **Fully from snapshots** (IOPV is in the file). Note A-share ETFs are T+0 with 0.001 tick and tight books, so this factor is well-defined.

---

## 15. Consolidated horizon & decay guide

| Factor | Best horizon(s) | Decay / half-life | Directional? | Snapshot | Tick |
|---|---|---|---|---|---|
| OFI (CKS) | 15s, 30s | seconds; impact tail minutes | **Yes** | Approx | **Exact** |
| Microprice | 15s | sub-sec → sec | Yes | **Yes** | Yes |
| Best-depth OIR | 15s–60s | sec → few min | Yes | **Yes** | Yes |
| Multi-level WDI | 15s–60s | sec → few min | Yes | **Yes** | Yes |
| Trade imbalance | 15s–60s (rev by 300s) | sec → min | Yes | Approx | **Exact** |
| Kyle λ | conditioning | slow regime | No (scaling) | Approx | Exact |
| Amihud | conditioning | slow regime | No (magnitude) | **Yes** | Yes |
| Spread decomp | conditioning | persistent | No (regime) | Approx | Exact |
| Book slope/shape | conditioning | quasi-stationary | No (vol) | **Yes** | Yes |
| Arrival/cancel/resiliency | 60s–900s | sec–min | Mixed | No | **Yes** |
| RV / BV / kernel / semi-var | all (magnitude) | slow/persistent | No (risk) | **Yes** | Yes |
| Trade duration (ACD) | conditioning | persistent | No (liquidity) | No | **Yes** |
| Queue position | 15s | sub-sec → sec | Yes | No | **Yes** |
| VPIN | regime/vol | long window | No (risk) | Approx | Exact |
| IOPV premium | 15s–900s | mean-reverting | **Yes** | **Yes** | n/a |

---

## References

- Amihud, Y. (2002). Illiquidity and Stock Returns: Cross-Section and Time-Series Effects. *Journal of Financial Markets*, 5(1), 31–56.
- Andersen, T. G., & Bondarenko, O. (2014). VPIN and the Flash Crash. *Journal of Financial Markets*. (critique: VPIN ≈ volume/volatility proxy)
- Barndorff-Nielsen, O. E., & Shephard, N. (2004). Power and Bipower Variation with Stochastic Volatility and Jumps. *Journal of Financial Econometrics*, 2(2), 1–37.
- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2008). Designing Realized Kernels to Measure the Ex Post Variation of Equity Returns in the Presence of Noise. *Econometrica*, 76(6), 1481–1536.
- Barndorff-Nielsen, O. E., Kinnebrock, S., & Shephard, N. (2010). Measuring Downside Risk — Realised Semivariance.
- Bouchaud, J.-P., Gefen, Y., Potters, M., & Wyart, M. (2004). Fluctuations and Response in Financial Markets. *Quantitative Finance*, 4(2), 176–190. (propagator, `G(t)∼t^{-1/2}`)
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

### Notes on sourcing and confidence
- Formulas for OFI, microprice, OIR, VPIN, Amihud, Roll, RV/BV/kernel, semi-variance, and ACD are standard and cross-checked; implement exactly as written.
- Quantitative magnitudes (OFI R² ~0.4–0.7 at second scales; A-share LOB alpha decaying within ~1–5s) are directional benchmarks from the literature — **re-estimate on this dataset**; decay/half-life numbers vary by instrument and liquidity.
- Some search-engine summaries cited secondary or unverifiable titles; those were excluded here. Only references I am confident are real are listed. Treat exact R²/half-life point estimates as approximate and validate empirically.
