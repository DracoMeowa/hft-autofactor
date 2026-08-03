// hftaf/book.cpp — snapshot-anchored top-10 book with tick updates.
#include "hftaf/book.hpp"
#include <algorithm>

namespace hftaf {

void BookState::reset() {
  for (auto& l : bids_) l = BookLevel{};
  for (auto& l : asks_) l = BookLevel{};
  synced_ = false;
  has_snapshot_ = false;
  resync_count_ = 0;
  last_snapshot_time_ = -1;
}

void BookState::apply_snapshot(const Snapshot& s) {
  if (has_snapshot_ && !synced_) ++resync_count_;   // snapshot restored a diverged book
  bids_ = s.bids;
  asks_ = s.asks;
  last_snapshot_time_ = s.time;
  has_snapshot_ = true;
  synced_ = true;                                    // snapshots are authoritative
}

bool BookState::synced() const { return synced_; }
bool BookState::has_snapshot() const { return has_snapshot_; }
std::uint64_t BookState::resync_count() const { return resync_count_; }
TsMs BookState::last_snapshot_time() const { return last_snapshot_time_; }

PriceI BookState::best_bid_price() const { return bids_[0].price; }
QtyI   BookState::best_bid_qty() const { return bids_[0].volume; }
PriceI BookState::best_ask_price() const { return asks_[0].price; }
QtyI   BookState::best_ask_qty() const { return asks_[0].volume; }
const std::array<BookLevel, 10>& BookState::bids() const { return bids_; }
const std::array<BookLevel, 10>& BookState::asks() const { return asks_; }

void BookState::consume_side(std::array<BookLevel, 10>& side, PriceI price, QtyI volume) {
  // Remove `volume` units at `price`, walking levels in best-to-worst order.
  QtyI remaining = volume;
  for (auto& lv : side) {
    if (lv.price == 0 || lv.volume <= 0) continue;
    if (lv.price != price) continue;
    const QtyI take = std::min(lv.volume, remaining);
    lv.volume -= take;
    remaining -= take;
    if (remaining == 0) break;
  }
  if (remaining > 0) {
    // Fill walked through depth that was not displayed (hidden liquidity,
    // swept levels we never saw, or missed events) => divergence.
    synced_ = false;
    return;
  }
  // Compact: pull non-empty levels towards index 0, preserving price priority.
  std::size_t w = 0;
  for (std::size_t r = 0; r < side.size(); ++r) {
    if (side[r].price > 0 && side[r].volume > 0) {
      if (w != r) side[w] = side[r];
      ++w;
    }
  }
  for (std::size_t k = w; k < side.size(); ++k) side[k] = BookLevel{};
}

void BookState::insert_level(std::array<BookLevel, 10>& side, bool is_bid, PriceI price, QtyI volume) {
  // Same-price level: add to it.
  for (auto& lv : side) {
    if (lv.price == price) {
      lv.volume += volume;
      lv.num_orders += 1;
      return;
    }
    if (lv.price == 0) break;   // reached the empty tail
  }
  // Determine the worst displayed level; drops beyond it are invisible to the
  // top-10 book and ignored.
  int worst = -1;
  for (int k = 9; k >= 0; --k) {
    if (side[k].price > 0) { worst = k; break; }
  }
  const bool better = is_bid ? (worst < 0 || price > side[worst].price)
                             : (worst < 0 || price < side[worst].price);
  if (worst == 9 && !better) return;   // book full and this is worse than all displayed

  // Insertion point: first level this order is better than (or the empty tail).
  int pos = -1;
  for (int k = 0; k < 10; ++k) {
    if (side[k].price == 0) { pos = k; break; }
    if (is_bid ? (price > side[k].price) : (price < side[k].price)) { pos = k; break; }
  }
  if (pos < 0) return;
  // Shift worse levels down by one (level 9 falls off).
  for (int k = 9; k > pos; --k) side[k] = side[k - 1];
  side[pos] = BookLevel{price, volume, 1};
}

void BookState::cancel_level(std::array<BookLevel, 10>& side, PriceI price, QtyI volume) {
  if (volume <= 0) return;   // cancel size unknown: cannot quantify, leave book as-is
  bool found = false;
  for (auto& lv : side) {
    if (lv.price == price && lv.volume > 0) {
      const QtyI take = std::min(lv.volume, volume);
      lv.volume -= take;
      volume -= take;
      found = true;
      if (lv.volume == 0) lv.num_orders = 0;
      if (volume == 0) break;
    }
  }
  // Cancelling depth that is no longer displayed (already consumed by trades)
  // is NOT a divergence; silently ignore when !found.
  if (!found) return;
  // Compact empty levels.
  std::size_t w = 0;
  for (std::size_t r = 0; r < side.size(); ++r) {
    if (side[r].price > 0 && side[r].volume > 0) {
      if (w != r) side[w] = side[r];
      ++w;
    }
  }
  for (std::size_t k = w; k < side.size(); ++k) side[k] = BookLevel{};
}

bool BookState::check_crossed() {
  if (bids_[0].price > 0 && asks_[0].price > 0 && bids_[0].price >= asks_[0].price) {
    synced_ = false;
    return true;
  }
  return false;
}

void BookState::apply_trade(const TickEvent& t) {
  if (!t.is_trade || !has_snapshot_ || !synced_) return;
  if (t.side == Side::Buy) {
    // Buyer aggressor consumes ask depth at the trade price.
    consume_side(asks_, t.price, t.volume);
  } else if (t.side == Side::Sell) {
    consume_side(bids_, t.price, t.volume);
  } else {
    // '-' print: aggressor side unattributable; leave book as-is (snapshot
    // re-anchors within ~3s). Not counted as divergence.
    return;
  }
  check_crossed();
}

void BookState::apply_order(const TickEvent& t) {
  if (t.is_trade || !has_snapshot_ || !synced_) return;
  const bool cxl = (t.ord_type == 'D' || t.ord_type == 'X');
  if (cxl) {
    if (t.side == Side::Buy) cancel_level(bids_, t.price, t.volume);
    else if (t.side == Side::Sell) cancel_level(asks_, t.price, t.volume);
  } else {
    if (t.side == Side::Buy) insert_level(bids_, true, t.price, t.volume);
    else if (t.side == Side::Sell) insert_level(asks_, false, t.price, t.volume);
  }
  check_crossed();
}

}  // namespace hftaf
