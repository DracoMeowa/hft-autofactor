// hftaf/decode.hpp — CSV parsing, header-driven schemas, tick/snapshot decode,
// cancel classification, and ETF universe filtering.
//
// All column indices are resolved BY NAME from the header row (never by
// position), so the engine is robust to column reordering across vendor dumps.
// Alias lists cover the SSE/SZSE naming variants documented in
// docs/data_dictionary.md.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include "hftaf/types.hpp"

namespace hftaf {

// Trim leading/trailing space/tab/CR (never any other byte) from a field view.
std::string_view trim(std::string_view s);

// Split `line` on ',' into `fields` (views into `line`; valid while `line`
// lives). L2 dumps carry no quoted/embedded commas; if that ever changes the
// data dictionary must be updated first. Returns the field count.
std::size_t split_csv(std::string_view line, std::vector<std::string_view>& fields);

// Like split_csv but stops scanning after `upto + 1` fields have been
// collected; the remainder of the line is never touched. Returns the number
// of fields collected (<= upto + 1). Used by the engine fast path to decide
// parse outcome / instrument identity on rows that are then skipped.
std::size_t split_csv_prefix(std::string_view line, std::vector<std::string_view>& fields, std::size_t upto);

// Exact fixed-point decimal -> milli-CNY. No float roundtrip: "3.456" -> 3456,
// "3.45" -> 3450, "3" -> 3000, "0.001" -> 1. Fractional digits beyond 3 are
// truncated (documented convention). Rejects empty/signed/malformed input.
bool parse_price_milli(std::string_view s, PriceI& out);

bool parse_int(std::string_view s, std::int64_t& out);

// "HHMMSSmmm" (9 digits) -> ms since midnight; also accepts "HHMMSS" (6 digits,
// seconds resolution) since some snapshot dumps omit the ms part.
bool parse_time_hhmmssmmm(std::string_view s, TsMs& out);

// Column indices resolved by NAME from the header row; -1 = absent.
struct TickSchema {
  int seq = -1;            // required: ordering authority (SeqNo / ApplSeqNum)
  int instrument = -1;     // required
  int trade2_order1 = -1;  // required: 1=order, 2=trade
  int transact_time = -1;  // required
  int price = -1;          // required
  int volume = -1;         // required
  int ord_side = -1;       // required: orders 1=buy 2=sell
  int ord_type = -1;       // required: cancel decode per data dictionary
  int trd_bs = -1;         // required: trade aggressor 'B'/'S'/'-'
  int trd_money = -1;
  int trd_buy_no = -1;
  int trd_sell_no = -1;
  int ord_no = -1;
  int biz_index = -1;
  int exchange_id = -1;
  int channel_no = -1;
  int trade_date = -1;
  int trans_flag = -1;
  int order_trd_volume = -1;
  int tick_status = -1;
};

struct SnapshotSchema {
  int instrument = -1;     // required
  int update_time = -1;    // required
  int last = -1;           // required
  int seq = -1;            // optional (SeqNo / ApplSeqNum if present)
  int pre_close = -1;
  int open_px = -1;
  int high = -1;
  int low = -1;
  int trade_volume = -1;   // cumulative since open
  int iopv = -1;
  int total_bid_vol = -1;
  int total_ask_vol = -1;
  std::array<int, 10> bid_px{}, bid_vol{}, ask_px{}, ask_vol{};
  std::array<int, 10> bid_orders{}, ask_orders{};
  SnapshotSchema() {
    for (int k = 0; k < 10; ++k) {
      bid_px[k] = bid_vol[k] = ask_px[k] = ask_vol[k] = -1;
      bid_orders[k] = ask_orders[k] = -1;
    }
    // Level 0 price+volume are required; validated in make_snapshot_schema.
  }
};

bool make_tick_schema(const std::vector<std::string_view>& header, TickSchema& out, std::string& err);
bool make_snapshot_schema(const std::vector<std::string_view>& header, SnapshotSchema& out, std::string& err);
bool parse_tick(const TickSchema&, std::string_view line, TickEvent& out, std::string& err);
// parse_tick over pre-split fields. Accept/reject semantics are IDENTICAL to
// parse_tick: all required columns are validated the same way, and optional
// trailing columns absent from `fields` keep their defaults exactly as when
// the column is missing from the row. A prefix split covering every required
// column therefore decides parse outcome without scanning the rest of the row.
bool parse_tick_fields(const TickSchema&, const std::vector<std::string_view>& fields, TickEvent& out, std::string& err);
bool parse_snapshot(const SnapshotSchema&, std::string_view line, Snapshot& out, std::string& err);

// Cancel classification per exchange; mapping documented in docs/data_dictionary.md.
// If decode is unreliable for an exchange these return false and engine logs a
// warning (cancel_ratio_60s / order_arrival_60s then emit NaN rather than wrong
// values).
bool sse_order_is_cancel(const TickEvent& t);
bool szse_order_is_cancel(const TickEvent& t);

// Reliability of the cancel decode above, per exchange. The engine warns (and
// the tick factors gated on cancels emit NaN) when this is false.
bool sse_cancel_decode_reliable();
bool szse_cancel_decode_reliable();
bool cancel_decode_reliable(const std::string& exchange);
bool order_is_cancel(const TickEvent& t, const std::string& exchange);

// v1 universe: ETFs only. SSE codes start 50/51/52/56/58, SZSE codes 15/16.
bool is_etf_code(const Symbol& s, const std::string& exchange);
// Same predicate on a raw (untrimmed) field view, for pre-parse filtering.
bool is_etf_code_sv(std::string_view code, const std::string& exchange);

}  // namespace hftaf
