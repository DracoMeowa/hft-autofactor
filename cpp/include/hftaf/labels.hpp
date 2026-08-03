// hftaf/labels.hpp — strictly future-only forward-return labels.
#pragma once
#include <functional>
#include <memory>
#include <vector>
#include "hftaf/types.hpp"
#include "hftaf/session.hpp"

namespace hftaf {

struct LabelConfig { std::vector<int> horizons_s = {15, 30, 60, 300, 900}; };

// STRICT FUTURE-ONLY labels. For observation t and horizon H:
//   fwd_mid[H]  = (mid(t+H) - mid(t)) / mid(t), mid from snapshot bid1/ask1 (two-sided required)
//   fwd_last[H] = (last(t+H) - last(t)) / last(t)
// where P(t+H) is taken at the FIRST snapshot of that instrument with time >= t+H inside the
// same continuous session. NaN when: window crosses lunch/close-auction/session-end, price at
// t invalid, or no usable snapshot before session end. NEVER pads/forward-fills.
// Rows are moved in with factor values attached and emitted complete (or flushed NaN at day end).
class LabelBuilder {
 public:
  explicit LabelBuilder(LabelConfig cfg, Session session);
  ~LabelBuilder();
  using Sink = std::function<void(Row&&)>;
  void set_sink(Sink sink);
  void push(Row&& observation, const Snapshot& s);   // per instrument, ascending time
  void end_instrument_day(const Symbol& inst);       // flush remaining rows (absent labels)
  void reset();                                       // new day/instrument context
  const LabelConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hftaf
