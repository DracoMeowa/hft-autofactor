// hftaf/types.hpp — core value types for the hft-autofactor engine.
//
// Determinism contract (see docs/architecture.md):
//   * Prices are int64 milli-CNY (1 = 0.001 CNY, the fund tick size).
//   * Quantities are int64 fund units. Money is raw int64 as printed.
//   * Time is int64 ms since 00:00:00 of the trading day (local).
//   * No floats in persisted market state; doubles only in derived factors.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace hftaf {

using PriceI = std::int64_t;  // milli-CNY (1 = 0.001 CNY); 0 = unset/invalid
using QtyI   = std::int64_t;  // fund units / shares
using MoneyI = std::int64_t;  // raw integer money as printed in feed
using TsMs   = std::int64_t;  // ms since 00:00:00 of the trading day (local)

struct Symbol {
  char data[12]{};
  std::uint8_t size = 0;
};

bool operator==(const Symbol& a, const Symbol& b);
bool operator<(const Symbol& a, const Symbol& b);   // lexicographic; used for output ordering
struct SymbolHash { std::size_t operator()(const Symbol& s) const; };

// Helpers (not part of the binding interface, used across the engine/tests).
Symbol make_symbol(const char* s, std::size_t len);
std::string symbol_to_string(const Symbol& s);

// --- inline definitions (header-only so no extra .cpp is needed) ---
inline bool operator==(const Symbol& a, const Symbol& b) {
  if (a.size != b.size) return false;
  return std::memcmp(a.data, b.data, a.size) == 0;
}
inline bool operator<(const Symbol& a, const Symbol& b) {
  const std::size_t n = a.size < b.size ? a.size : b.size;
  const int c = std::memcmp(a.data, b.data, n);
  if (c != 0) return c < 0;
  return a.size < b.size;
}
inline std::size_t SymbolHash::operator()(const Symbol& s) const {
  // FNV-1a 64-bit.
  std::size_t h = 1469598103934665603ULL;
  for (std::uint8_t i = 0; i < s.size; ++i) {
    h ^= static_cast<unsigned char>(s.data[i]);
    h *= 1099511628211ULL;
  }
  return h;
}
inline Symbol make_symbol(const char* s, std::size_t len) {
  Symbol sym;
  sym.size = static_cast<std::uint8_t>(len > 12 ? 12 : len);
  for (std::uint8_t i = 0; i < sym.size; ++i) sym.data[i] = s[i];
  return sym;
}
inline std::string symbol_to_string(const Symbol& s) { return std::string(s.data, s.size); }

struct BookLevel { PriceI price = 0; QtyI volume = 0; std::int32_t num_orders = 0; };

enum class Side : std::uint8_t { None = 0, Buy = 1, Sell = 2 };

struct TickEvent {                 // one row of 1_channel_N.csv.gz
  TsMs time = 0;                   // TransactTime
  std::int64_t seq = 0;            // SeqNo (per-channel ordering authority)
  Symbol instrument;               // InstrumentID
  bool is_trade = false;           // Trade2_Order1: 1=order, 2=trade
  Side side = Side::None;          // order: OrdSide; trade: aggressor from TrdBSFlag ('B'=>Buy)
  PriceI price = 0;
  QtyI volume = 0;
  MoneyI trd_money = 0;            // TrdMoney (trades only)
  char ord_type = 0;               // orders: 'A' etc., decode per docs/data_dictionary.md
  char trd_bs = 0;                 // trades: 'B' | 'S' | '-'
  std::int64_t ord_no = 0;
  std::int64_t trd_buy_no = 0;
  std::int64_t trd_sell_no = 0;
  std::int64_t biz_index = 0;
};

struct Snapshot {                  // one row of 1_snapshot.csv.gz
  TsMs time = 0;                   // UpdateTime
  std::int64_t seq = -1;           // -1 if file lacks a sequence column
  Symbol instrument;
  PriceI last = 0, pre_close = 0, open_px = 0, high = 0, low = 0;
  PriceI iopv = 0;                 // 0 => not provided
  bool iopv_valid = false;
  QtyI cum_trade_volume = 0;       // cumulative since open; engine computes deltas, resets on decrease
  QtyI total_bid_vol = 0, total_ask_vol = 0;
  std::array<BookLevel, 10> bids{}, asks{};  // index 0 = best
};

struct Row {                       // one output CSV row (factor sample + labels)
  std::string date;                // YYYYMMDD
  std::string exchange;            // "sse" | "szse"
  Symbol instrument;
  TsMs time = 0;
  std::int64_t snap_seq = -1;
  std::uint32_t flags = 0;         // bit0 BOOK_UNSYNCED, bit1 SEQ_GAP_BEFORE, bit2 IOPV_INVALID, bit3 ONE_SIDED_BOOK
  // market state for downstream backtest
  double mid_px = 0, last_px = 0, bid1_px = 0, ask1_px = 0;   // CNY (milli/1000)
  QtyI bid1_qty = 0, ask1_qty = 0, depth_bid5 = 0, depth_ask5 = 0;
  std::vector<double> factors;     // registry order; NaN => absent (warm-up/invalid)
  std::vector<double> fwd_mid;     // per horizon; NaN => ABSENT (never padded)
  std::vector<double> fwd_last;    // per horizon; NaN => ABSENT
};

// Flag bit positions (see docs/data_dictionary.md).
enum FlagBits : std::uint32_t {
  FLAG_BOOK_UNSYNCED   = 1u << 0,  // tick-updated book diverged before this snapshot re-anchor
  FLAG_SEQ_GAP_BEFORE  = 1u << 1,  // per-channel SeqNo gap observed since previous snapshot
  FLAG_IOPV_INVALID    = 1u << 2,  // IOPV missing/non-positive in this snapshot
  FLAG_ONE_SIDED_BOOK  = 1u << 3,  // bid1 or ask1 missing (price == 0)
};

}  // namespace hftaf
