// hftaf/labels.cpp — LabelBuilder: strictly future-only forward-return labels.
//
// ABSENT semantics: labels are NaN (empty CSV cell) whenever the window
// crosses lunch/close-auction/session-end, the base price at t is invalid, or
// no usable snapshot exists before session end. NEVER pads/forward-fills.
#include "hftaf/labels.hpp"
#include <cmath>
#include <deque>
#include <unordered_map>

namespace hftaf {

struct LabelBuilder::Impl {
  LabelConfig cfg;
  Session session;
  Sink sink;

  struct Pending {
    Row row;                        // carries factor values; fwd_* resolved in place
    TsMs t = 0;
    double mid_t = 0.0, last_t = 0.0;
    bool mid_ok = false, last_ok = false;
    std::vector<char> fits;         // per horizon: window fits the continuous session
    std::vector<char> done_mid, done_last;
  };
  std::unordered_map<Symbol, std::deque<Pending>, SymbolHash> pend;

  bool all_done(const Pending& p) const {
    for (std::size_t i = 0; i < cfg.horizons_s.size(); ++i)
      if (!p.done_mid[i] || !p.done_last[i]) return false;
    return true;
  }
};

LabelBuilder::LabelBuilder(LabelConfig cfg, Session session)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = std::move(cfg);
  impl_->session = session;
}

LabelBuilder::~LabelBuilder() = default;

void LabelBuilder::set_sink(Sink sink) { impl_->sink = std::move(sink); }
const LabelConfig& LabelBuilder::config() const { return impl_->cfg; }

void LabelBuilder::push(Row&& observation, const Snapshot& s) {
  Impl& im = *impl_;
  const Symbol inst = observation.instrument;
  const TsMs U = s.time;

  // Resolution source derived from this snapshot.
  const BookLevel& b1 = s.bids[0];
  const BookLevel& a1 = s.asks[0];
  const bool mid_u_ok = b1.price > 0 && a1.price > 0;
  const double mid_u = mid_u_ok ? (static_cast<double>(b1.price) + a1.price) / 2000.0 : 0.0;
  const bool last_u_ok = s.last > 0;
  const double last_u = last_u_ok ? s.last / 1000.0 : 0.0;

  auto& dq = im.pend[inst];

  // 1) Resolve pending observations against this snapshot (ascending U, so the
  //    first snapshot satisfying U >= t+H is exactly this one).
  for (auto& p : dq) {
    for (std::size_t i = 0; i < im.cfg.horizons_s.size(); ++i) {
      const TsMs H = static_cast<TsMs>(im.cfg.horizons_s[i]) * 1000;
      if (!p.fits[i]) {
        // Window crosses lunch/auction/session-end => ABSENT.
        p.row.fwd_mid[i] = std::nan("");
        p.row.fwd_last[i] = std::nan("");
        p.done_mid[i] = p.done_last[i] = 1;
        continue;
      }
      if (U >= p.t + H) {
        p.row.fwd_mid[i] = (p.mid_ok && mid_u_ok) ? (mid_u - p.mid_t) / p.mid_t : std::nan("");
        p.row.fwd_last[i] = (p.last_ok && last_u_ok) ? (last_u - p.last_t) / p.last_t : std::nan("");
        p.done_mid[i] = p.done_last[i] = 1;
      }
    }
  }
  // 2) Emit fully-resolved rows from the front (times stay ascending).
  while (!dq.empty() && im.all_done(dq.front())) {
    if (im.sink) im.sink(std::move(dq.front().row));
    dq.pop_front();
  }

  // 3) Append the new observation.
  Impl::Pending p;
  p.row = std::move(observation);
  p.t = U;
  p.mid_t = p.row.mid_px;
  p.last_t = p.row.last_px;
  p.mid_ok = std::isfinite(p.mid_t) && p.mid_t > 0;
  p.last_ok = std::isfinite(p.last_t) && p.last_t > 0;
  const std::size_t nh = im.cfg.horizons_s.size();
  p.row.fwd_mid.assign(nh, std::nan(""));
  p.row.fwd_last.assign(nh, std::nan(""));
  p.fits.resize(nh);
  p.done_mid.assign(nh, 0);
  p.done_last.assign(nh, 0);
  for (std::size_t i = 0; i < nh; ++i) {
    p.fits[i] = horizon_fits_session(im.session, U, static_cast<TsMs>(im.cfg.horizons_s[i]) * 1000) ? 1 : 0;
  }
  dq.push_back(std::move(p));
}

void LabelBuilder::end_instrument_day(const Symbol& inst) {
  Impl& im = *impl_;
  auto it = im.pend.find(inst);
  if (it == im.pend.end()) return;
  // Flush remaining rows; unresolved labels stay NaN (ABSENT).
  for (auto& p : it->second) {
    if (im.sink) im.sink(std::move(p.row));
  }
  im.pend.erase(it);
}

void LabelBuilder::reset() { impl_->pend.clear(); }

}  // namespace hftaf
