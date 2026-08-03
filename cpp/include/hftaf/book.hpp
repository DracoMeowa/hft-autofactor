// hftaf/book.hpp — snapshot-anchored top-10 order book per instrument.
#pragma once
#include <array>
#include <cstdint>
#include "hftaf/types.hpp"

namespace hftaf {

// Snapshot-anchored top-10 book. Authoritative state = last snapshot; between
// snapshots: trades consume aggressed-side depth (side from TrdBSFlag,
// exchange-provided — no Lee-Ready inference), limit adds insert at price,
// cancels remove best-effort per decode. On inconsistency the book is marked
// unsynced until the next snapshot re-anchors it. Consumers check synced()
// and rows carry FLAG_BOOK_UNSYNCED for the affected interval; flagged
// intervals are excluded from evaluation downstream, never imputed.
class BookState {
 public:
  void reset();
  void apply_snapshot(const Snapshot& s);
  void apply_trade(const TickEvent& t);     // requires t.is_trade
  void apply_order(const TickEvent& t);     // requires !t.is_trade
  bool synced() const;
  bool has_snapshot() const;                // at least one snapshot anchored
  std::uint64_t resync_count() const;       // snapshots that restored sync after divergence
  PriceI best_bid_price() const; QtyI best_bid_qty() const;
  PriceI best_ask_price() const; QtyI best_ask_qty() const;
  const std::array<BookLevel, 10>& bids() const;
  const std::array<BookLevel, 10>& asks() const;
  TsMs last_snapshot_time() const;

 private:
  void consume_side(std::array<BookLevel, 10>& side, PriceI price, QtyI volume);
  void insert_level(std::array<BookLevel, 10>& side, bool is_bid, PriceI price, QtyI volume);
  void cancel_level(std::array<BookLevel, 10>& side, PriceI price, QtyI volume);
  bool check_crossed();

  std::array<BookLevel, 10> bids_{};
  std::array<BookLevel, 10> asks_{};
  bool synced_ = false;
  bool has_snapshot_ = false;
  std::uint64_t resync_count_ = 0;
  TsMs last_snapshot_time_ = -1;
};

}  // namespace hftaf
