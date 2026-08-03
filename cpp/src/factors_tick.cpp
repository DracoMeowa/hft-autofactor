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
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>

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
// ofi_60s — Cont-Kukanov-Stoikov OFI on event-updated best quotes, normalized
// by mean best depth. Per event:
//   eB = 1{bp_n >= bp_{n-1}} bq_n - 1{bp_n <= bp_{n-1}} bq_{n-1}
//   eA = 1{ap_n <= ap_{n-1}} aq_n - 1{ap_n >= ap_{n-1}} aq_{n-1}
//   contrib = eB - eA
// value = sum(contrib over trailing 60s) / mean(bq+aq over those events).
// Quotes are re-based at every snapshot (authoritative re-anchor); events while
// the book is unsynced contribute nothing (prev quote invalidated).
// ---------------------------------------------------------------------------
class OFI60s final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "ofi_60s"; return n; }

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
    while (!dq.empty() && dq.front().ts < t.time - kWindowMs) dq.pop_front();

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
    while (!dq.empty() && dq.front().ts < s.time - kWindowMs) dq.pop_front();

    auto fit = first_tick_.find(s.instrument);
    if (fit == first_tick_.end() || s.time - fit->second < kWindowMs || dq.empty())
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
  std::unordered_map<Symbol, std::deque<Event>, SymbolHash> events_;
  std::unordered_map<Symbol, Quote, SymbolHash> prev_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_tick_;
};

// ---------------------------------------------------------------------------
// trade_imbalance_60s = (Vbuy - Vsell)/(Vbuy + Vsell) via exchange aggressor
// flags (TrdBSFlag); '-' prints excluded (unattributable).
// ---------------------------------------------------------------------------
class TradeImbalance60s final : public TickFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "trade_imbalance_60s"; return n; }

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
    while (!dq.empty() && dq.front().ts < t.time - kWindowMs) dq.pop_front();
  }

  void on_snapshot(const Snapshot& s, const BookState&) override {
    auto& dq = events_[s.instrument];
    while (!dq.empty() && dq.front().ts < s.time - kWindowMs) dq.pop_front();
    auto fit = first_trade_.find(s.instrument);
    if (fit == first_trade_.end() || s.time - fit->second < kWindowMs)
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
  if (name == "ofi_60s") return std::make_unique<OFI60s>();
  if (name == "trade_imbalance_60s") return std::make_unique<TradeImbalance60s>();
  if (name == "order_arrival_60s") return std::make_unique<OrderArrival60s>();
  if (name == "cancel_ratio_60s") return std::make_unique<CancelRatio60s>();
  if (name == "future_mid_15s") return std::make_unique<FutureMid15s>();
  if (name == "future_trade_sign") return std::make_unique<FutureTradeSign>();
  return nullptr;
}

}  // namespace hftaf
