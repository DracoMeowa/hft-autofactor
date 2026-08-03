// hftaf/factors_snapshot.cpp — snapshot-family factors.
//
// All of these compute from the snapshot-anchored book at the snapshot's own
// timestamp, so they are trivially causal: value(inst) uses only information
// available at availability time <= t. Per-instrument state is keyed by Symbol
// and accumulations iterate in fixed (level / window) order, so output is
// bitwise reproducible run-to-run.
#include "hftaf/factors.hpp"
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <unordered_map>

namespace hftaf {

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

double milli_to_cny(PriceI p) { return static_cast<double>(p) / 1000.0; }

// Best bid/ask from the anchored book (post apply_snapshot).
void best_quotes(const BookState& book, PriceI& bp, QtyI& bq, PriceI& ap, QtyI& aq) {
  bp = book.best_bid_price(); bq = book.best_bid_qty();
  ap = book.best_ask_price(); aq = book.best_ask_qty();
}
bool two_sided(PriceI bp, PriceI ap) { return bp > 0 && ap > 0; }

// Base for stateless snapshot-native factors: store last value per instrument.
class SnapshotFactorBase : public IFactor {
 public:
  void on_day_start(const FactorContext& ctx) override { ctx_ = ctx; last_.clear(); }
  void on_instrument_day_start(const Symbol& inst) override { last_.erase(inst); }
  void on_tick(const TickEvent&, const BookState&) override {}
  bool value(const Symbol& inst, double& out) const override {
    auto it = last_.find(inst);
    if (it == last_.end() || std::isnan(it->second)) return false;
    out = it->second;
    return true;
  }

 protected:
  void store(const Symbol& inst, double v) { last_[inst] = v; }
  FactorContext ctx_;
  std::unordered_map<Symbol, double, SymbolHash> last_;
};

// quoted_spread_ticks = (ask1 - bid1) in ticks (tick = 1 milli-CNY = 0.001 CNY).
class QuotedSpreadTicks final : public SnapshotFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "quoted_spread_ticks"; return n; }
  void on_snapshot(const Snapshot& s, const BookState& book) override {
    PriceI bp, ap; QtyI bq, aq;
    best_quotes(book, bp, bq, ap, aq);
    if (!two_sided(bp, ap)) return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(ap - bp));   // ticks are 1 milli-CNY
  }
};

// microprice_dev = ((ask1*bidQ + bid1*askQ)/(bidQ+askQ) - mid), bps of mid.
class MicropriceDev final : public SnapshotFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "microprice_dev"; return n; }
  void on_snapshot(const Snapshot& s, const BookState& book) override {
    PriceI bp, ap; QtyI bq, aq;
    best_quotes(book, bp, bq, ap, aq);
    if (!two_sided(bp, ap) || bq + aq <= 0) return store(s.instrument, kNan);
    const double mid = (static_cast<double>(bp) + static_cast<double>(ap)) * 0.5;
    if (mid <= 0) return store(s.instrument, kNan);
    // Microprice weights each side by the OPPOSITE queue depth.
    const double micro_num = static_cast<double>(ap) * static_cast<double>(bq) +
                             static_cast<double>(bp) * static_cast<double>(aq);
    const double micro = micro_num / static_cast<double>(bq + aq);
    const double dev_bps = (micro - mid) / mid * 1e4;
    store(s.instrument, dev_bps);
  }
};

// oir = (bidQ - askQ)/(bidQ + askQ) at the best level.
class OIR final : public SnapshotFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "oir"; return n; }
  void on_snapshot(const Snapshot& s, const BookState& book) override {
    PriceI bp, ap; QtyI bq, aq;
    best_quotes(book, bp, bq, ap, aq);
    if (!two_sided(bp, ap)) return store(s.instrument, kNan);
    const QtyI denom = bq + aq;
    if (denom <= 0) return store(s.instrument, kNan);
    store(s.instrument, static_cast<double>(bq - aq) / static_cast<double>(denom));
  }
};

// wdi = exp(-k/2)-weighted 5-level depth imbalance normalized to [-1,1].
class WDI final : public SnapshotFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "wdi"; return n; }
  void on_snapshot(const Snapshot& s, const BookState& book) override {
    PriceI bp, ap; QtyI bq, aq;
    best_quotes(book, bp, bq, ap, aq);
    if (!two_sided(bp, ap)) return store(s.instrument, kNan);
    const auto& bids = book.bids();
    const auto& asks = book.asks();
    double num = 0.0, denom = 0.0;
    for (int k = 0; k < 5; ++k) {
      const double w = std::exp(-static_cast<double>(k) * 0.5);
      const QtyI bv = bids[k].price > 0 ? bids[k].volume : 0;
      const QtyI av = asks[k].price > 0 ? asks[k].volume : 0;
      num += w * static_cast<double>(bv - av);
      denom += w * static_cast<double>(bv + av);
    }
    if (denom <= 0) return store(s.instrument, kNan);
    double v = num / denom;
    if (v > 1.0) v = 1.0; else if (v < -1.0) v = -1.0;
    store(s.instrument, v);
  }
};

// book_slope = mean OLS slope of ln(cum depth) vs distance-from-mid (bp), levels 0..4.
class BookSlope final : public SnapshotFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "book_slope"; return n; }
  void on_snapshot(const Snapshot& s, const BookState& book) override {
    PriceI bp, ap; QtyI bq, aq;
    best_quotes(book, bp, bq, ap, aq);
    if (!two_sided(bp, ap)) return store(s.instrument, kNan);
    const double mid = (static_cast<double>(bp) + static_cast<double>(ap)) * 0.5;
    if (mid <= 0) return store(s.instrument, kNan);
    double sb = 0.0, sa = 0.0;
    if (!side_slope(book.bids(), mid, true, sb)) return store(s.instrument, kNan);
    if (!side_slope(book.asks(), mid, false, sa)) return store(s.instrument, kNan);
    store(s.instrument, 0.5 * (sb + sa));
  }

 private:
  // OLS of ln(cum depth) on distance-from-mid (bps) across levels 0..4.
  static bool side_slope(const std::array<BookLevel, 10>& side, double mid, bool is_bid, double& out) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    QtyI cum = 0;
    for (int k = 0; k < 5; ++k) {
      if (side[k].price <= 0 || side[k].volume <= 0) break;   // contiguous from best
      cum += side[k].volume;
      const double dist_bp = is_bid
          ? (mid - static_cast<double>(side[k].price)) / mid * 1e4
          : (static_cast<double>(side[k].price) - mid) / mid * 1e4;
      const double y = std::log(static_cast<double>(cum));
      sx += dist_bp; sy += y; sxx += dist_bp * dist_bp; sxy += dist_bp * y;
      ++n;
    }
    if (n < 2) return false;
    const double denom = static_cast<double>(n) * sxx - sx * sx;
    if (std::fabs(denom) < 1e-12) return false;   // degenerate (all same distance)
    out = (static_cast<double>(n) * sxy - sx * sy) / denom;
    return true;
  }
};

// iopv_premium = (last - iopv)/iopv in bps; NaN unless iopv_valid && two-sided book.
class IopvPremium final : public SnapshotFactorBase {
 public:
  const std::string& name() const override { static const std::string n = "iopv_premium"; return n; }
  void on_snapshot(const Snapshot& s, const BookState& book) override {
    PriceI bp, ap; QtyI bq, aq;
    best_quotes(book, bp, bq, ap, aq);
    if (!two_sided(bp, ap) || !s.iopv_valid || s.iopv <= 0 || s.last <= 0)
      return store(s.instrument, kNan);
    const double prem_bps = (static_cast<double>(s.last) - static_cast<double>(s.iopv)) /
                            static_cast<double>(s.iopv) * 1e4;
    store(s.instrument, prem_bps);
  }
};

// rv_60s / rv_300s = sqrt(sum of squared 3s log-mid returns) over the trailing
// window. We sample mid at each snapshot; consecutive-snapshot log returns are
// accumulated in fixed order.
//
// WARM-UP CONTRACT (must match the Python differential reference): the value is
// emitted only once (t - first_snapshot_time) >= window AND the window contains
// at least 80% of the nominal return count (window/3s). Otherwise empty cell.
class RealizedVol final : public SnapshotFactorBase {
 public:
  explicit RealizedVol(int window_s)
      : window_ms_(static_cast<TsMs>(window_s) * 1000),
        min_returns_(static_cast<int>((window_ms_ / 3000) * 4 / 5)) {
    name_ = "rv_" + std::to_string(window_s) + "s";
  }
  const std::string& name() const override { return name_; }
  void on_instrument_day_start(const Symbol& inst) override {
    SnapshotFactorBase::on_instrument_day_start(inst);
    mids_.erase(inst);
    first_ts_.erase(inst);
  }
  void on_snapshot(const Snapshot& s, const BookState& book) override {
    PriceI bp, ap; QtyI bq, aq;
    best_quotes(book, bp, bq, ap, aq);
    auto& dq = mids_[s.instrument];
    if (first_ts_.find(s.instrument) == first_ts_.end()) first_ts_[s.instrument] = s.time;
    if (!two_sided(bp, ap)) {
      // One-sided: no mid this snapshot; break the return chain by recording a gap.
      dq.push_back({s.time, kNan});
    } else {
      const double mid = (static_cast<double>(bp) + static_cast<double>(ap)) * 0.5;
      dq.push_back({s.time, std::log(mid)});
    }
    // Trim old entries beyond the window.
    while (!dq.empty() && dq.front().t < s.time - window_ms_) dq.pop_front();
    // Sum squared log returns over consecutive usable snapshots (fixed order).
    double sum = 0.0;
    int ret_count = 0;
    bool prev_valid = false;
    double prev_log = 0.0;
    for (const auto& e : dq) {
      if (std::isnan(e.log_mid)) { prev_valid = false; continue; }
      if (prev_valid) {
        const double r = e.log_mid - prev_log;
        sum += r * r;
        ++ret_count;
      }
      prev_log = e.log_mid;
      prev_valid = true;
    }
    const bool warm = (s.time - first_ts_[s.instrument]) >= window_ms_ && ret_count >= min_returns_;
    store(s.instrument, warm ? std::sqrt(sum) : kNan);
  }

 private:
  struct MidPt { TsMs t; double log_mid; };
  TsMs window_ms_;
  int min_returns_;
  std::string name_;
  std::unordered_map<Symbol, std::deque<MidPt>, SymbolHash> mids_;
  std::unordered_map<Symbol, TsMs, SymbolHash> first_ts_;
};

}  // namespace

std::unique_ptr<IFactor> make_snapshot_factor(const std::string& name) {
  if (name == "quoted_spread_ticks") return std::make_unique<QuotedSpreadTicks>();
  if (name == "microprice_dev") return std::make_unique<MicropriceDev>();
  if (name == "oir") return std::make_unique<OIR>();
  if (name == "wdi") return std::make_unique<WDI>();
  if (name == "book_slope") return std::make_unique<BookSlope>();
  if (name == "iopv_premium") return std::make_unique<IopvPremium>();
  if (name == "rv_60s") return std::make_unique<RealizedVol>(60);
  if (name == "rv_300s") return std::make_unique<RealizedVol>(300);
  return nullptr;
}

}  // namespace hftaf
