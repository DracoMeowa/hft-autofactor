// hftaf/factors_tick.cpp — tick-family factors (causal 60s rolling windows
// sampled at snapshot times) + the deliberate look-ahead canaries.
//
// Causality: events only enter a window once their tick has been processed;
// windows are pruned by event time; values sampled at snapshot time t use only
// events with TransactTime <= t (merge rule: those ticks precede snapshot t).
//
// Warm-up contract (must match the Python differential reference): a 60s
// window emits a value only once (t - first_relevant_event_time) >= 60s.
//
// Cancel gating: order_arrival_60s and cancel_ratio_60s emit NaN on exchanges
// whose cancel decode is unreliable (see decode.cpp / docs/data_dictionary.md).
#include "hftaf/factors.hpp"
#include "hftaf/decode.hpp"  // order_is_cancel / cancel_decode_reliable
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace hftaf {

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr TsMs kWindowMs = 60000;        // all v1 tick factors: trailing 60s
constexpr TsMs kCanaryHorizonMs = 15000; // canaries peek 15s into the future
// Retention for the trade-sign canary: delayed rows finalize at their
// instrument's NEXT snapshot after t+15s, so keep ample slack (150s) beyond
// the horizon to survive sparse snapshot cadence. Memory stays bounded.
constexpr TsMs kCanaryTradeKeepMs = 10 * kCanaryHorizonMs;

// ---------------------------------------------------------------------------
// Base for causal tick-window factors.
// ---------------------------------------------------------------------------
class TickFactorBase : public IFactor {
 public:
  void on_day_start(const FactorContext& ctx) override {
    ctx_ = ctx;
    cancel_reliable_ = cancel_decode_reliable(ctx.exchange);
    last_.clear();
  }
  void on_instrument_day_start(const Symbol& inst) override { last_.erase(inst); }
  bool value(const Symbol& inst, double& out) const override {
    auto it = last_.find(inst);
    if (it == last_.end() || std::isnan(it->second)) return false;
    out = it->second;
    return true;
  }

 protected:
  void store(const Symbol& inst, double v) { last_[inst] = v; }
  FactorContext ctx_;
  bool cancel_reliable_ = false;
  std::unordered_map<Symbol, double, SymbolHash> last_;
};

// ---------------------------------------------------------------------------
// ofi_60s / ofi_30s / ofi_15s — Cont-Kukanov-Stoikov OFI on event-updated
// best quotes, normalized by mean best depth, over a parameterized trailing
// window (60s is the v1 default; 30/15s added for the iter-003 wide-table
// expansion, same formula and warm-up contract). Per event:
//   eB = 1{bp_n >= bp_{n-1}} bq_n - 1{bp_n <= bp_{n-1}} bq_{n-1}
//   eA = 1{ap_n <= ap_{n-1}} aq_n - 1{ap_n >= ap_{n-1}} aq_{n-1}
//   contrib = eB - eA
// value = sum(contrib over trailing window) / mean(bq+aq over those events).
// Quotes are re-based at every snapshot (authoritative re-anchor); events while
// the book is unsynced contribute nothing (prev quote invalidated).
// ---------------------------------------------------------------------------
class OFIWindow final : public TickFactorBase {
 public:
  explicit OFIWindow(int window_s)
      : window_ms_(static_cast<TsMs>(window_s) * 1000) {
    name_ = "ofi_" + std::to_string(window_s) + "s";
  }
  const std::string& name() const override { return name_; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    events_.erase(inst);
    prev_.erase(inst);
    first_tick_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState& book) override {
    if (first_tick_.find(t.instrument) == first_tick_.end()) first_tick_[t.instrument] = t.time;
    auto& dq = events_[t.instrument];
    // Prune by event time to keep the window bounded even between snapshots.
    while (!dq.empty() && dq.front().ts < t.time - window_ms_) dq.pop_front();

    if (!book.synced()) { prev_[t.instrument].valid = false; return; }
    Quote cur;
    cur.bp = book.best_bid_price(); cur.bq = book.best_bid_qty();
    cur.ap = book.best_ask_price(); cur.aq = book.best_ask_qty();
    cur.valid = cur.bp > 0 && cur.ap > 0;
    if (!cur.valid) { prev_[t.instrument] = cur; return; }

    auto pit = prev_.find(t.instrument);
    if (pit != prev_.end() && pit->second.valid) {
      const Quote& p = pit->second;
      const QtyI eb = (cur.bp >= p.bp ? cur.bq : 0) - (cur.bp <= p.bp ? p.bq : 0);
      const QtyI ea = (cur.ap <= p.ap ? cur.aq : 0) - (cur.ap >= p.ap ? p.aq : 0);
      dq.push_back(Event{t.time, eb - ea, cur.bq + cur.aq});
    }
    prev_[t.instrument] = cur;
  }

  void on_snapshot(const Snapshot& s, const BookState& book) override {
    // Re-base the quote tracker at the authoritative snapshot state.
    Quote q;
    q.bp = book.best_bid_price(); q.bq = book.best_bid_qty();
    q.ap = book.best_ask_price(); q.aq = book.best_ask_qty();
    q.valid = book.synced() && q.bp > 0 && q.ap > 0;
    prev_[s.instrument] = q;

    auto& dq = events_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - window_ms_) dq.pop_front();

    auto fit = first_tick_.find(s.instrument);
    if (fit == first_tick_.end() || s.time - fit->second < window_ms_ || dq.empty())
      return store(s.instrument, kNan);

    QtyI sum_contrib = 0, sum_depth = 0;
    std::int64_t n = 0;
    for (const auto& e : dq) {           // fixed (arrival) order accumulation
      sum_contrib += e.contrib;
      sum_depth += e.depth;
      ++n;
    }
    if (n <= 0 || sum_depth <= 0) return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(sum_contrib) /
                        (static_cast<double>(sum_depth) / static_cast<double>(n)));
  }

 private:
  struct Quote { PriceI bp = 0, ap = 0; QtyI bq = 0, aq = 0; bool valid = false; };
  struct Event { TsMs ts; QtyI contrib; QtyI depth; };
  TsMs window_ms_;
  std::string name_;
  std::unordered_map<Symbol, std::deque<Event>, SymbolHash> events_;
  std::unordered_map<Symbol, Quote, SymbolHash> prev_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_tick_;
};

// ---------------------------------------------------------------------------
// trade_imbalance_60s / _30s / _15s = (Vbuy - Vsell)/(Vbuy + Vsell) via
// exchange aggressor flags (TrdBSFlag) over a parameterized trailing window;
// '-' prints excluded (unattributable). 60s is the v1 default; 30/15s added
// for the iter-003 wide-table expansion with the same formula and warm-up.
// ---------------------------------------------------------------------------
class TradeImbalanceWindow final : public TickFactorBase {
 public:
  explicit TradeImbalanceWindow(int window_s)
      : window_ms_(static_cast<TsMs>(window_s) * 1000) {
    name_ = "trade_imbalance_" + std::to_string(window_s) + "s";
  }
  const std::string& name() const override { return name_; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    events_.erase(inst);
    first_trade_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (!t.is_trade) return;
    QtyI sv = 0;
    if (t.trd_bs == 'B') sv = t.volume;
    else if (t.trd_bs == 'S') sv = -t.volume;
    else return;                          // '-' print: excluded
    if (first_trade_.find(t.instrument) == first_trade_.end())
      first_trade_[t.instrument] = t.time;
    auto& dq = events_[t.instrument];
    dq.push_back(Event{t.time, sv});
    while (!dq.empty() && dq.front().ts < t.time - window_ms_) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto& dq = events_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - window_ms_) dq.pop_front();
    auto fit = first_trade_.find(s.instrument);
    if (fit == first_trade_.end() || s.time - fit->second < window_ms_)
      return store(s.instrument, kNan);
    QtyI vb = 0, vs = 0;
    for (const auto& e : dq) {
      if (e.signed_vol >= 0) vb += e.signed_vol;
      else vs -= e.signed_vol;
    }
    if (vb + vs <= 0) return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(vb - vs) / static_cast<double>(vb + vs));
  }

 private:
  struct Event { TsMs ts; QtyI signed_vol; };
  TsMs window_ms_;
  std::string name_;
  std::unordered_map<Symbol, std::deque<Event>, SymbolHash> events_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_trade_;
};

// ---------------------------------------------------------------------------
// order_arrival_60s = signed imbalance of NEW-order arrival counts (cancels
// excluded): (N_buy - N_sell)/(N_buy + N_sell) over trailing 60s.
// NaN when the exchange's cancel decode is unreliable.
// ---------------------------------------------------------------------------
class OrderArrival60s final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "order_arrival_60s"; return n; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    events_.erase(inst);
    first_order_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (t.is_trade) return;
    if (first_order_.find(t.instrument) == first_order_.end())
      first_order_[t.instrument] = t.time;
    if (!cancel_reliable_) return;        // cannot separate adds from cancels
    if (order_is_cancel(t, ctx_.exchange)) return;
    if (t.side != Side::Buy && t.side != Side::Sell) return;
    auto& dq = events_[t.instrument];
    dq.push_back(Event{t.time, t.side});
    while (!dq.empty() && dq.front().ts < t.time - kWindowMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    if (!cancel_reliable_) return store(s.instrument, kNan);
    auto& dq = events_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - kWindowMs) dq.pop_front();
    auto fit = first_order_.find(s.instrument);
    if (fit == first_order_.end() || s.time - fit->second < kWindowMs)
      return store(s.instrument, kNan);
    std::int64_t nb = 0, ns = 0;
    for (const auto& e : dq) {
      if (e.side == Side::Buy) ++nb; else ++ns;
    }
    if (nb + ns <= 0) return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(nb - ns) / static_cast<double>(nb + ns));
  }

 private:
  struct Event { TsMs ts; Side side; };
  std::unordered_map<Symbol, std::deque<Event>, SymbolHash> events_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_order_;
};

// ---------------------------------------------------------------------------
// cancel_ratio_60s = n_cancel / (n_cancel + n_add) over trailing 60s.
// NaN when the exchange's cancel decode is unreliable.
// ---------------------------------------------------------------------------
class CancelRatio60s final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "cancel_ratio_60s"; return n; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    events_.erase(inst);
    first_order_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (t.is_trade) return;
    if (first_order_.find(t.instrument) == first_order_.end())
      first_order_[t.instrument] = t.time;
    if (!cancel_reliable_) return;
    auto& dq = events_[t.instrument];
    dq.push_back(Event{t.time, order_is_cancel(t, ctx_.exchange)});
    while (!dq.empty() && dq.front().ts < t.time - kWindowMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    if (!cancel_reliable_) return store(s.instrument, kNan);
    auto& dq = events_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - kWindowMs) dq.pop_front();
    auto fit = first_order_.find(s.instrument);
    if (fit == first_order_.end() || s.time - fit->second < kWindowMs)
      return store(s.instrument, kNan);
    std::int64_t nc = 0, na = 0;
    for (const auto& e : dq) {
      if (e.is_cancel) ++nc; else ++na;
    }
    if (nc + na <= 0) return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(nc) / static_cast<double>(nc + na));
  }

 private:
  struct Event { TsMs ts; bool is_cancel; };
  std::unordered_map<Symbol, std::deque<Event>, SymbolHash> events_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_order_;
};

// ---------------------------------------------------------------------------
// Trade-size / trade-arrival family (wishlist materialization, opt-in via
// --factors; NOT in kDefaultFactorNames so already-produced runs keep their
// skip-if-done sidecars valid).
//
// Shared base: causal trailing-60s deque of ALL trades. '-' prints (aggressor
// side unattributable) are INCLUDED here: volume and timestamp are known and
// these statistics are side-blind by design (the signed analogues are
// ofi_60s / trade_imbalance_60s). Trades with volume <= 0 are malformed and
// skipped. Warm-up follows the v1 contract: a window value is emitted only
// once (t - first_trade_time) >= 60s and the window is non-empty.
// ---------------------------------------------------------------------------
class TradeWindow60sBase : public TickFactorBase {
 public:
  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    trades_.erase(inst);
    first_trade_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (!t.is_trade || t.volume <= 0) return;
    if (first_trade_.find(t.instrument) == first_trade_.end())
      first_trade_[t.instrument] = t.time;
    auto& dq = trades_[t.instrument];
    dq.push_back(Trd{t.time, t.volume});
    while (!dq.empty() && dq.front().ts < t.time - kWindowMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto& dq = trades_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - kWindowMs) dq.pop_front();
    auto fit = first_trade_.find(s.instrument);
    if (fit == first_trade_.end() || s.time - fit->second < kWindowMs || dq.empty())
      return store(s.instrument, kNan);
    store(s.instrument, compute(dq));
  }

 protected:
  struct Trd { TsMs ts; QtyI volume; };
  virtual double compute(const std::deque<Trd>& dq) const = 0;
  std::unordered_map<Symbol, std::deque<Trd>, SymbolHash> trades_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_trade_;
};

// avg_trade_size_60s = mean per-trade volume (fund units) over trailing 60s.
class AvgTradeSize60s final : public TradeWindow60sBase {
 public:
  const std::string& name() const override { static const std::string n = "avg_trade_size_60s"; return n; }

 protected:
  double compute(const std::deque<Trd>& dq) const override {
    QtyI total = 0;
    for (const auto& e : dq) total += e.volume;
    if (total <= 0) return kNan;
    return static_cast<double>(total) / static_cast<double>(dq.size());
  }
};

// n_trades_60s = trade count over trailing 60s (arrival rate; the raw input
// for burst-vs-baseline prototypes such as trade_arrival_burst).
class NTrades60s final : public TradeWindow60sBase {
 public:
  const std::string& name() const override { static const std::string n = "n_trades_60s"; return n; }

 protected:
  double compute(const std::deque<Trd>& dq) const override {
    return static_cast<double>(dq.size());
  }
};

// large_trade_share_60s = share of trailing-60s volume carried by the
// largest ceil(n/10) trades (at least 1): a self-normalizing measure of how
// concentrated flow is in large prints (Kyle 1985; Bouchaud et al. 2004
// square-root impact => large trades carry disproportionate information).
// The quantile cut is endogenous to the same trailing window, so the column
// is causal and comparable across instruments without a fixed size threshold.
// The top-k volume sum is tie-invariant: equal sizes contribute equally no
// matter which of them land beyond the cut.
class LargeTradeShare60s final : public TradeWindow60sBase {
 public:
  const std::string& name() const override { static const std::string n = "large_trade_share_60s"; return n; }

 protected:
  double compute(const std::deque<Trd>& dq) const override {
    std::vector<QtyI> sizes;
    sizes.reserve(dq.size());
    QtyI total = 0;
    for (const auto& e : dq) { sizes.push_back(e.volume); total += e.volume; }
    if (total <= 0) return kNan;
    const std::size_t n = sizes.size();
    const std::size_t k_large = std::max<std::size_t>(1, (n + 9) / 10);  // ceil(n/10)
    std::nth_element(sizes.begin(), sizes.begin() + (n - k_large), sizes.end());
    QtyI big = 0;
    for (std::size_t i = n - k_large; i < n; ++i) big += sizes[i];
    return static_cast<double>(big) / static_cast<double>(total);
  }
};

// trade_gap_ms = ms elapsed since the most recent trade, sampled at snapshot
// time (inter-trade duration; small <=> high arrival rate). Instantaneous
// statistic: defined from the first trade onward, no 60s warm-up. No rows are
// emitted across the lunch break, and liquid instruments resume trading
// within the skew window of the first afternoon snapshot, so the break itself
// rarely surfaces as a large gap here (quiet resumes still show it). Clamped
// at >= 0: snapshot-phase skew (see on_snapshot) can make the newest known
// trade's stamp slightly later than the snapshot's own UpdateTime; the trade
// is already known, so the honest elapsed time is "just traded" = 0.
class TradeGapMs final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "trade_gap_ms"; return n; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    last_trade_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (!t.is_trade) return;
    last_trade_[t.instrument] = t.time;
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto it = last_trade_.find(s.instrument);
    if (it == last_trade_.end()) return store(s.instrument, kNan);
    // Clamp at zero. The merged snapshot stream carries per-instrument
    // UpdateTime phase skew within each publication batch (one SSE batch's
    // stamps span ~1s; the drainers are typically NON-ETF barrier snapshots
    // that never emit rows). The engine's shared merge cursor drains every
    // tick with time <= U (U = UpdateTime of the snapshot being processed),
    // so such a barrier can pull trades stamped slightly AFTER this
    // instrument's own snapshot time into factor state before its
    // on_snapshot runs (observed on 20250702: 16,022 negative rows, worst
    // -970 ms, concentrated in the open burst). At the row's availability
    // time the trade is already known, so the honest elapsed time is "just
    // traded" => 0, never < 0.
    const TsMs dt = s.time - it->second;
    store(s.instrument, static_cast<double>(dt > 0 ? dt : 0));
  }

 private:
  std::unordered_map<Symbol, TsMs, SymbolHash> last_trade_;
};

// ---------------------------------------------------------------------------
// iter-003 wide-table expansion (#140/#144): short-window OFI/imbalance live
// in the parameterized classes above; new columns below. All opt-in via
// --factors (NOT in kDefaultFactorNames).
// ---------------------------------------------------------------------------

// buy_vol_60s / sell_vol_60s = total aggressor-attributed trade volume on one
// side over the trailing 60s (fund units). '-' prints (unattributable) count
// for neither side. Warm-up follows the signed-trade family: emitted only
// once (t - first signed trade time) >= 60s. Raw level (zero is a legitimate
// value once warm); the explore lane derives ratios / z-scores.
class SideVol60s final : public TickFactorBase {
 public:
  explicit SideVol60s(Side side) : side_(side) {
    name_ = (side == Side::Buy) ? "buy_vol_60s" : "sell_vol_60s";
  }
  const std::string& name() const override { return name_; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    events_.erase(inst);
    first_trade_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (!t.is_trade || t.trd_bs == '-') return;
    if (first_trade_.find(t.instrument) == first_trade_.end())
      first_trade_[t.instrument] = t.time;
    if ((t.trd_bs == 'B') != (side_ == Side::Buy)) return;
    auto& dq = events_[t.instrument];
    dq.push_back(Event{t.time, t.volume});
    while (!dq.empty() && dq.front().ts < t.time - kWindowMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto& dq = events_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - kWindowMs) dq.pop_front();
    auto fit = first_trade_.find(s.instrument);
    if (fit == first_trade_.end() || s.time - fit->second < kWindowMs)
      return store(s.instrument, kNan);
    QtyI total = 0;
    for (const auto& e : dq) total += e.volume;
    store(s.instrument, static_cast<double>(total));
  }

 private:
  struct Event { TsMs ts; QtyI volume; };
  Side side_;
  std::string name_;
  std::unordered_map<Symbol, std::deque<Event>, SymbolHash> events_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_trade_;
};

// large_trade_net_share_60s = net signed volume share of the largest trades:
// among the signed ('B'/'S'; '-' excluded) trades in the trailing 60s, take
// the largest k = ceil(n/10) (>= 1) by size and emit
//   sum(their signed volume) / sum(|volume| over all signed trades) in [-1,1].
// Directional companion to large_trade_share_60s (institutional activity
// probe). Deterministic under size ties: stable sort keeps arrival order, so
// equal-size prints cannot flip the signed sum.
class LargeTradeNetShare60s final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "large_trade_net_share_60s"; return n; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    trades_.erase(inst);
    first_trade_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (!t.is_trade || t.volume <= 0) return;
    QtyI sv = 0;
    if (t.trd_bs == 'B') sv = t.volume;
    else if (t.trd_bs == 'S') sv = -t.volume;
    else return;                          // '-' print: excluded
    if (first_trade_.find(t.instrument) == first_trade_.end())
      first_trade_[t.instrument] = t.time;
    auto& dq = trades_[t.instrument];
    dq.push_back(Trd{t.time, sv});
    while (!dq.empty() && dq.front().ts < t.time - kWindowMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto& dq = trades_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - kWindowMs) dq.pop_front();
    auto fit = first_trade_.find(s.instrument);
    if (fit == first_trade_.end() || s.time - fit->second < kWindowMs || dq.empty())
      return store(s.instrument, kNan);
    std::vector<Trd> v(dq.begin(), dq.end());
    std::stable_sort(v.begin(), v.end(), [](const Trd& a, const Trd& b) {
      const QtyI aa = a.signed_vol >= 0 ? a.signed_vol : -a.signed_vol;
      const QtyI ab = b.signed_vol >= 0 ? b.signed_vol : -b.signed_vol;
      return aa > ab;
    });
    const std::size_t n = v.size();
    const std::size_t k_large = std::max<std::size_t>(1, (n + 9) / 10);  // ceil(n/10)
    QtyI top_signed = 0, total_abs = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const QtyI av = v[i].signed_vol >= 0 ? v[i].signed_vol : -v[i].signed_vol;
      total_abs += av;
      if (i < k_large) top_signed += v[i].signed_vol;
    }
    if (total_abs <= 0) return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(top_signed) / static_cast<double>(total_abs));
  }

 private:
  struct Trd { TsMs ts; QtyI signed_vol; };
  std::unordered_map<Symbol, std::deque<Trd>, SymbolHash> trades_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_trade_;
};

// book_event_intensity_60s = count of feed events (orders + trades) touching
// the instrument over the trailing 60s, per second: information-arrival rate
// decoupled from the 3s snapshot cadence (Hawkes self-excitation proxy).
// Every event counts regardless of add/cancel classification: on SSE cancels
// cannot be decoded reliably, but all events are book-touching by definition.
// Warm-up: 60s from the first event of the day.
class BookEventIntensity60s final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "book_event_intensity_60s"; return n; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    events_.erase(inst);
    first_event_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (first_event_.find(t.instrument) == first_event_.end())
      first_event_[t.instrument] = t.time;
    auto& dq = events_[t.instrument];
    dq.push_back(t.time);
    while (!dq.empty() && dq.front() < t.time - kWindowMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto& dq = events_[s.instrument];
    while (!dq.empty() && dq.front() < s.time - kWindowMs) dq.pop_front();
    auto fit = first_event_.find(s.instrument);
    if (fit == first_event_.end() || s.time - fit->second < kWindowMs)
      return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(dq.size()) /
                        (static_cast<double>(kWindowMs) / 1000.0));
  }

 private:
  std::unordered_map<Symbol, std::deque<TsMs>, SymbolHash> events_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_event_;
};

// ---------------------------------------------------------------------------
// CANARIES — deliberate look-ahead, shipped only behind --canaries.
// The engine finalizes canary rows kCanaryHorizonMs late (value_at), so these
// factors can illegitimately see the future. The mask test MUST fail when
// these columns are present; that failure proves the validator works.
// ---------------------------------------------------------------------------

// future_mid_15s: mid of the first snapshot at/after t + 15s.
class FutureMid15s final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "future_mid_15s"; return n; }
  bool is_canary() const override { return true; }

  // Snapshot-only factor: ticks carry no information for it (no-op override).
  void on_tick(const TickEvent&, const BookState&) override {}

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    snaps_.erase(inst);
  }

  void on_snapshot(const Snapshot& s, const BookState& book) override {
    auto& dq = snaps_[s.instrument];
    const PriceI bp = book.best_bid_price(), ap = book.best_ask_price();
    const bool ok = bp > 0 && ap > 0;
    const double mid = ok ? (static_cast<double>(bp) + static_cast<double>(ap)) / 2000.0 : kNan;
    dq.push_back(Point{s.time, mid, ok});
    // No pruning: snapshot cadence per instrument is unbounded, and a delayed
    // row may legitimately ask for any point after its creation. Snapshot
    // count per day is small (~10^4), so keep the full day.
  }

  bool value_at(const Symbol& inst, TsMs t, double& out) const override {
    auto it = snaps_.find(inst);
    if (it == snaps_.end()) return false;
    const TsMs target = t + kCanaryHorizonMs;
    for (const auto& p : it->second) {
      if (p.ts >= target) {
        if (!p.ok) return false;
        out = p.mid;
        return true;
      }
    }
    return false;   // future beyond end of data => absent
  }

 private:
  struct Point { TsMs ts; double mid; bool ok; };
  std::unordered_map<Symbol, std::deque<Point>, SymbolHash> snaps_;
};

// future_trade_sign: sign of net aggressor-signed trade volume in (t, t+15s].
class FutureTradeSign final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "future_trade_sign"; return n; }
  bool is_canary() const override { return true; }

  void on_instrument_day_start(const Symbol& inst) override {
    TickFactorBase::on_instrument_day_start(inst);
    trades_.erase(inst);
  }

  void on_tick(const TickEvent& t, const BookState&) override {
    if (!t.is_trade) return;
    QtyI sv = 0;
    if (t.trd_bs == 'B') sv = t.volume;
    else if (t.trd_bs == 'S') sv = -t.volume;
    else return;
    auto& dq = trades_[t.instrument];
    dq.push_back(Trd{t.time, sv});
    while (!dq.empty() && dq.front().ts < t.time - kCanaryTradeKeepMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto& dq = trades_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - kCanaryTradeKeepMs) dq.pop_front();
  }

  bool value_at(const Symbol& inst, TsMs t, double& out) const override {
    auto it = trades_.find(inst);
    if (it == trades_.end()) return false;
    QtyI sum = 0;
    bool any = false;
    for (const auto& e : it->second) {
      if (e.ts > t && e.ts <= t + kCanaryHorizonMs) { sum += e.signed_vol; any = true; }
    }
    if (!any) return false;
    out = (sum > 0) ? 1.0 : (sum < 0) ? -1.0 : 0.0;
    return true;
  }

 private:
  struct Trd { TsMs ts; QtyI signed_vol; };
  std::unordered_map<Symbol, std::deque<Trd>, SymbolHash> trades_;
};

}  // namespace

std::unique_ptr<IFactor> make_tick_factor(const std::string& name) {
  if (name == "ofi_60s") return std::make_unique<OFIWindow>(60);
  if (name == "ofi_30s") return std::make_unique<OFIWindow>(30);
  if (name == "ofi_15s") return std::make_unique<OFIWindow>(15);
  if (name == "trade_imbalance_60s") return std::make_unique<TradeImbalanceWindow>(60);
  if (name == "trade_imbalance_30s") return std::make_unique<TradeImbalanceWindow>(30);
  if (name == "trade_imbalance_15s") return std::make_unique<TradeImbalanceWindow>(15);
  if (name == "order_arrival_60s") return std::make_unique<OrderArrival60s>();
  if (name == "cancel_ratio_60s") return std::make_unique<CancelRatio60s>();
  if (name == "avg_trade_size_60s") return std::make_unique<AvgTradeSize60s>();
  if (name == "n_trades_60s") return std::make_unique<NTrades60s>();
  if (name == "large_trade_share_60s") return std::make_unique<LargeTradeShare60s>();
  if (name == "large_trade_net_share_60s") return std::make_unique<LargeTradeNetShare60s>();
  if (name == "buy_vol_60s") return std::make_unique<SideVol60s>(Side::Buy);
  if (name == "sell_vol_60s") return std::make_unique<SideVol60s>(Side::Sell);
  if (name == "book_event_intensity_60s") return std::make_unique<BookEventIntensity60s>();
  if (name == "trade_gap_ms") return std::make_unique<TradeGapMs>();
  if (name == "future_mid_15s") return std::make_unique<FutureMid15s>();
  if (name == "future_trade_sign") return std::make_unique<FutureTradeSign>();
  return nullptr;
}

}  // namespace hftaf
