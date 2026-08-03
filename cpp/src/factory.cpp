// hftaf/factory.cpp — registry construction: name -> factor instance.
#include "hftaf/factors.hpp"
#include <stdexcept>

namespace hftaf {

const std::vector<std::string> kDefaultFactorNames = {
    // snapshot family (CSV column order)
    "quoted_spread_ticks",
    "microprice_dev",
    "oir",
    "wdi",
    "book_slope",
    "iopv_premium",
    "rv_60s",
    "rv_300s",
    // tick family
    "ofi_60s",
    "trade_imbalance_60s",
    "order_arrival_60s",
    "cancel_ratio_60s",
};

namespace {
const std::vector<std::string> kCanaryNames = {
    "future_mid_15s",
    "future_trade_sign",
};

std::unique_ptr<IFactor> make_one(const std::string& name) {
  if (auto f = make_snapshot_factor(name)) return f;
  if (auto f = make_tick_factor(name)) return f;
  return nullptr;
}
}  // namespace

std::vector<std::unique_ptr<IFactor>> make_default_registry() {
  return make_registry({}, false);
}

std::vector<std::unique_ptr<IFactor>> make_registry(const std::vector<std::string>& names,
                                                    bool include_canaries) {
  std::vector<std::unique_ptr<IFactor>> reg;
  const std::vector<std::string>& wanted = names.empty() ? kDefaultFactorNames : names;
  for (const auto& n : wanted) {
    auto f = make_one(n);
    if (!f) throw std::invalid_argument("unknown factor name: " + n);
    reg.push_back(std::move(f));
  }
  if (include_canaries) {
    for (const auto& n : kCanaryNames) {
      // Do not duplicate if the caller already requested a canary explicitly.
      bool present = false;
      for (const auto& f : reg) {
        if (f->name() == n) { present = true; break; }
      }
      if (present) continue;
      auto f = make_one(n);
      if (!f) throw std::invalid_argument("unknown canary factor name: " + n);
      reg.push_back(std::move(f));
    }
  }
  return reg;
}

}  // namespace hftaf
