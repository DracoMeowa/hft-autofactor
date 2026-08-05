// hftaf/factors.hpp — IFactor interface, factor registry, canonical v1 factors.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "hftaf/book.hpp"
#include "hftaf/session.hpp"
#include "hftaf/types.hpp"

namespace hftaf {

struct FactorContext { std::string date; std::string exchange; Session session; };

// CAUSALITY CONTRACT: value(inst) at snapshot time t may use only information
// with availability time <= t. Return false (=> empty CSV cell) during warm-up
// or when inputs are invalid. Per-instrument state lives inside the factor
// (map keyed by Symbol).
//
// on_snapshot is called after BookState::apply_snapshot; on_tick is called
// after the event was applied to the book, so `book` is the post-event state.
class IFactor {
 public:
  virtual ~IFactor() = default;
  virtual const std::string& name() const = 0;
  virtual void on_day_start(const FactorContext& ctx) = 0;
  virtual void on_instrument_day_start(const Symbol& inst) = 0;
  virtual void on_snapshot(const Snapshot& s, const BookState& book) = 0;
  virtual void on_tick(const TickEvent& t, const BookState& book) = 0;
  virtual bool value(const Symbol& inst, double& out) const = 0;

  // Point-in-time query for a SPECIFIC past snapshot time `t`. Only the
  // deliberate look-ahead canaries override this (the engine finalizes their
  // rows late so future data is visible); causal factors use value() and the
  // default returns false => empty cell.
  virtual bool value_at(const Symbol& inst, TsMs t, double& out) const {
    (void)inst; (void)t; (void)out;
    return false;
  }

  // True only for the deliberate look-ahead canaries shipped behind
  // --canaries. The engine samples these with a delayed attach so they can
  // (illegally) see future data; the mask test MUST fail when they are
  // present, proving the validator detects leakage.
  virtual bool is_canary() const { return false; }
};

// v1 registry — names are the CSV column names, in this order. Formulas/citations in
// docs/knowledge/microstructure_factors.md; exact normalization constants in docs/architecture.md.
//  snapshot family: quoted_spread_ticks  (ask1-bid1 in ticks)
//                   microprice_dev       ((ask1*bidQ+bid1*askQ)/(bidQ+aq))/(bidQ+askQ) - mid), bps of mid
//                   oir                  (bidQ-askQ)/(bidQ+askQ) at best level
//                   wdi                  exp(-k/2)-weighted 5-level depth imbalance, normalized to [-1,1]
//                   book_slope           mean OLS slope of ln(cum depth) vs distance-from-mid (bp), levels 0..4
//                   iopv_premium         (last-iopv)/iopv in bps; NaN unless iopv_valid && two-sided book
//                   rv_60s, rv_300s      sqrt(sum of squared 3s log-mid returns) over trailing window
//  tick family:     ofi_60s              Cont-Kukanov-Stoikov OFI on event-updated best quotes / mean best depth
//                   trade_imbalance_60s  (Vbuy-Vsell)/(Vbuy+Vsell) via TrdBSFlag ('-' prints excluded)
//                   order_arrival_60s    signed imbalance of new-order arrival counts (cancels excluded)
//                   cancel_ratio_60s     n_cancel/(n_cancel+n_add) over trailing 60s
//  canaries (only with --canaries): future_mid_15s, future_trade_sign  (deliberate look-ahead;
//  mask test MUST fail when these are present)
//
//  OPT-IN wishlist columns (buildable via --factors but NOT in kDefaultFactorNames,
//  so already-produced runs keep their skip-if-done sidecars valid; materialized
//  per instrument on demand — see docs/roadmap/panel-columns-wishlist.md):
//   snapshot family: cum_trade_vol       cumulative-since-open TradeVolume pass-through
//                                          (NaN on intra-day decrease => feed anomaly)
//                    total_bid_vol / total_ask_vol  full-book volume totals (raw units)
//                    bid_orders5 / ask_orders5      sum of NumOrders over levels 0..4
//                    open_px / high_px / low_px / pre_close_px  intraday reference
//                                          prices in CNY (milli/1000)
//                    iopv_velocity        IOPV change rate over trailing 60s (bps/s)
//   tick family:     avg_trade_size_60s      mean per-trade volume over trailing 60s
//                    n_trades_60s            trade count over trailing 60s (arrival rate)
//                    large_trade_share_60s   volume share of the largest ceil(n/10) trades
//                    large_trade_net_share_60s  signed net share of the largest ceil(n/10)
//                                          trades, in [-1,1] ('-' prints excluded)
//                    trade_gap_ms            ms since the most recent trade (no 60s warm-up;
//                                              clamped at 0 under snapshot-phase skew)
//                    ofi_15s / ofi_30s       short-window OFI (same formula as ofi_60s)
//                    trade_imbalance_15s / trade_imbalance_30s  short-window imbalance
//                    buy_vol_60s / sell_vol_60s  side-attributed trade volume (raw units;
//                                          '-' prints excluded; zero legitimate once warm)
//                    book_event_intensity_60s  feed events (orders+trades) per second
std::vector<std::unique_ptr<IFactor>> make_default_registry();
std::vector<std::unique_ptr<IFactor>> make_registry(const std::vector<std::string>& names, bool include_canaries = false);

// Ordered names of the default registry (CSV column order without canaries).
extern const std::vector<std::string> kDefaultFactorNames;

// Per-TU single-factor factories used by factory.cpp (additions to the
// binding interface; concrete classes stay private to their .cpp files).
// Return nullptr when `name` is not implemented in that translation unit.
std::unique_ptr<IFactor> make_snapshot_factor(const std::string& name);   // factors_snapshot.cpp
std::unique_ptr<IFactor> make_tick_factor(const std::string& name);       // factors_tick.cpp (incl. canaries)

}  // namespace hftaf
