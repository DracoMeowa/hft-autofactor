// hftaf/decode.cpp — CSV split, exact fixed-point/int/time parsers,
// header-driven schemas, tick/snapshot decode, cancel classification.
#include "hftaf/decode.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

namespace hftaf {

namespace {

std::string_view trim(std::string_view s) {
  std::size_t b = 0, e = s.size();
  while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) ++b;
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
  return s.substr(b, e - b);
}

// Case-insensitive header-name match (some dumps vary case).
bool name_eq(std::string_view a, const char* b) {
  std::size_t n = std::strlen(b);
  if (a.size() != n) return false;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::toupper(static_cast<unsigned char>(a[i])) !=
        std::toupper(static_cast<unsigned char>(b[i])))
      return false;
  }
  return true;
}

// Find the first header column whose name matches any alias; -1 if none.
int find_col(const std::vector<std::string_view>& header, std::initializer_list<const char*> aliases) {
  for (std::size_t i = 0; i < header.size(); ++i) {
    std::string_view h = trim(header[i]);
    for (const char* a : aliases) {
      if (name_eq(h, a)) return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

std::size_t split_csv(std::string_view line, std::vector<std::string_view>& fields) {
  fields.clear();
  std::size_t start = 0;
  for (std::size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      fields.push_back(line.substr(start, i - start));
      start = i + 1;
    }
  }
  return fields.size();
}

bool parse_price_milli(std::string_view s, PriceI& out) {
  s = trim(s);
  if (s.empty()) return false;
  std::size_t i = 0;
  if (s[0] == '+') i = 1;             // unsigned only; '-' rejected by digit check
  std::int64_t ipart = 0;
  bool any_digit = false;
  for (; i < s.size() && s[i] != '.'; ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    if (ipart > (INT64_MAX / 10000)) return false;
    ipart = ipart * 10 + (s[i] - '0');
    any_digit = true;
  }
  if (!any_digit) return false;
  std::int64_t frac = 0;
  int ndig = 0;
  if (i < s.size() && s[i] == '.') {
    ++i;
    for (; i < s.size(); ++i) {
      if (s[i] < '0' || s[i] > '9') return false;
      if (ndig < 3) {                 // keep first 3 fractional digits, truncate rest
        frac = frac * 10 + (s[i] - '0');
        ++ndig;
      }
    }
  }
  for (int k = ndig; k < 3; ++k) frac *= 10;   // pad ".45" -> 450 milli
  out = ipart * 1000 + frac;
  return true;
}

bool parse_int(std::string_view s, std::int64_t& out) {
  s = trim(s);
  if (s.empty()) return false;
  std::size_t i = 0;
  bool neg = false;
  if (s[0] == '+' || s[0] == '-') { neg = (s[0] == '-'); i = 1; }
  if (i >= s.size()) return false;
  std::int64_t v = 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    if (v > INT64_MAX / 10) return false;
    v = v * 10 + (s[i] - '0');
  }
  out = neg ? -v : v;
  return true;
}

bool parse_time_hhmmssmmm(std::string_view s, TsMs& out) {
  s = trim(s);
  auto digits_ok = [&](std::size_t n) {
    if (s.size() < n) return false;
    for (std::size_t i = 0; i < n; ++i)
      if (s[i] < '0' || s[i] > '9') return false;
    return true;
  };
  if (s.size() == 9 && digits_ok(9)) {
    int hh = (s[0] - '0') * 10 + (s[1] - '0');
    int mm = (s[2] - '0') * 10 + (s[3] - '0');
    int ss = (s[4] - '0') * 10 + (s[5] - '0');
    int ms = (s[6] - '0') * 100 + (s[7] - '0') * 10 + (s[8] - '0');
    if (hh > 23 || mm > 59 || ss > 59) return false;
    out = ((hh * 60 + mm) * 60 + ss) * 1000LL + ms;
    return true;
  }
  if (s.size() == 6 && digits_ok(6)) {
    int hh = (s[0] - '0') * 10 + (s[1] - '0');
    int mm = (s[2] - '0') * 10 + (s[3] - '0');
    int ss = (s[4] - '0') * 10 + (s[5] - '0');
    if (hh > 23 || mm > 59 || ss > 59) return false;
    out = ((hh * 60 + mm) * 60 + ss) * 1000LL;
    return true;
  }
  return false;
}

bool make_tick_schema(const std::vector<std::string_view>& header, TickSchema& out, std::string& err) {
  out = TickSchema{};
  out.seq            = find_col(header, {"SeqNo", "ApplSeqNum", "SequenceNo"});
  out.instrument     = find_col(header, {"InstrumentID", "SecCode", "Instrument", "Symbol"});
  out.trade2_order1  = find_col(header, {"Trade2_Order1", "TradeOrderFlag", "OrderType_TradeFlag"});
  out.transact_time  = find_col(header, {"TransactTime", "TransTime", "OrderTime", "TradeTime"});
  out.price          = find_col(header, {"Price"});
  out.volume         = find_col(header, {"Volume", "Qty"});
  out.ord_side       = find_col(header, {"OrdSide", "OrderSide", "Side"});
  out.ord_type       = find_col(header, {"OrdType", "OrderType"});
  out.trd_bs         = find_col(header, {"TrdBSFlag", "BSFlag", "TradeBSFlag"});
  out.trd_money      = find_col(header, {"TrdMoney"});
  out.trd_buy_no     = find_col(header, {"TrdBuyNo"});
  out.trd_sell_no    = find_col(header, {"TrdSellNo"});
  out.ord_no         = find_col(header, {"OrdNo"});
  out.biz_index      = find_col(header, {"BizIndex"});
  out.exchange_id    = find_col(header, {"ExchangeID"});
  out.channel_no     = find_col(header, {"ChannelNo"});
  out.trade_date     = find_col(header, {"TradeDate"});
  out.trans_flag     = find_col(header, {"TransFlag"});
  out.order_trd_volume = find_col(header, {"OrderTrdVolume"});
  out.tick_status    = find_col(header, {"TickStatus"});

  const char* missing[16];
  int n = 0;
  auto req = [&](int v, const char* nm) { if (v < 0) missing[n++] = nm; };
  req(out.seq, "SeqNo");
  req(out.instrument, "InstrumentID");
  req(out.trade2_order1, "Trade2_Order1");
  req(out.transact_time, "TransactTime");
  req(out.price, "Price");
  req(out.volume, "Volume");
  req(out.ord_side, "OrdSide");
  req(out.ord_type, "OrdType");
  req(out.trd_bs, "TrdBSFlag");
  if (n > 0) {
    err = "tick schema missing required column(s):";
    for (int i = 0; i < n; ++i) { err += ' '; err += missing[i]; }
    return false;
  }
  return true;
}

bool make_snapshot_schema(const std::vector<std::string_view>& header, SnapshotSchema& out, std::string& err) {
  out = SnapshotSchema{};
  out.instrument   = find_col(header, {"InstrumentID", "SecCode", "Instrument", "Symbol"});
  out.update_time  = find_col(header, {"UpdateTime", "SnapshotTime"});
  out.last         = find_col(header, {"LastPrice", "Last"});
  out.seq          = find_col(header, {"SeqNo", "ApplSeqNum", "SnapshotSeqNo"});
  out.pre_close    = find_col(header, {"PreClosePrice", "PreClose"});
  out.open_px      = find_col(header, {"OpenPrice", "Open"});
  out.high         = find_col(header, {"HighPrice", "High"});
  out.low          = find_col(header, {"LowPrice", "Low"});
  out.trade_volume = find_col(header, {"TradeVolume", "TotalVolume", "CumVolume"});
  out.iopv         = find_col(header, {"IOPV", "IOPVPrice"});
  out.total_bid_vol = find_col(header, {"TotalBidVolume"});
  out.total_ask_vol = find_col(header, {"TotalAskVolume"});

  for (int k = 0; k < 10; ++k) {
    char bp[16], bv[16], ap[16], av[16], bo[24], ao[24];
    std::snprintf(bp, sizeof(bp), "BidPrice%d", k);
    std::snprintf(bv, sizeof(bv), "BidVolume%d", k);
    std::snprintf(ap, sizeof(ap), "AskPrice%d", k);
    std::snprintf(av, sizeof(av), "AskVolume%d", k);
    std::snprintf(bo, sizeof(bo), "BidNumOrders%d", k);
    std::snprintf(ao, sizeof(ao), "AskNumOrders%d", k);
    out.bid_px[k] = find_col(header, {bp});
    out.bid_vol[k] = find_col(header, {bv});
    out.ask_px[k] = find_col(header, {ap});
    out.ask_vol[k] = find_col(header, {av});
    out.bid_orders[k] = find_col(header, {bo});
    out.ask_orders[k] = find_col(header, {ao});
    // Levels are contiguous: stop resolving once a level's price column is gone.
    if (out.bid_px[k] < 0) break;
  }

  const char* missing[16];
  int n = 0;
  auto req = [&](int v, const char* nm) { if (v < 0) missing[n++] = nm; };
  req(out.instrument, "InstrumentID");
  req(out.update_time, "UpdateTime");
  req(out.last, "LastPrice");
  req(out.bid_px[0], "BidPrice0");
  req(out.bid_vol[0], "BidVolume0");
  req(out.ask_px[0], "AskPrice0");
  req(out.ask_vol[0], "AskVolume0");
  if (n > 0) {
    err = "snapshot schema missing required column(s):";
    for (int i = 0; i < n; ++i) { err += ' '; err += missing[i]; }
    return false;
  }
  return true;
}

namespace {

bool field_str(const std::vector<std::string_view>& f, int idx, std::string_view& out) {
  if (idx < 0 || static_cast<std::size_t>(idx) >= f.size()) return false;
  out = f[idx];
  return true;
}

// Optional int: absent/empty => 0 / keep default, no error.
void opt_int(const std::vector<std::string_view>& f, int idx, std::int64_t& out) {
  std::string_view v;
  if (!field_str(f, idx, v)) return;
  v = trim(v);
  if (v.empty()) return;
  parse_int(v, out);
}

void opt_price(const std::vector<std::string_view>& f, int idx, PriceI& out) {
  std::string_view v;
  if (!field_str(f, idx, v)) return;
  v = trim(v);
  if (v.empty()) return;
  parse_price_milli(v, out);
}

}  // namespace

bool parse_tick(const TickSchema& sc, std::string_view line, TickEvent& out, std::string& err) {
  std::vector<std::string_view> f;
  split_csv(line, f);
  out = TickEvent{};

  std::string_view v;
  // Required fields.
  if (!field_str(f, sc.seq, v) || !parse_int(trim(v), out.seq)) { err = "bad SeqNo"; return false; }
  if (!field_str(f, sc.instrument, v)) { err = "missing InstrumentID"; return false; }
  v = trim(v);
  if (v.empty() || v.size() > 12) { err = "bad InstrumentID"; return false; }
  out.instrument = make_symbol(v.data(), v.size());

  int64_t t2o1 = 0;
  if (!field_str(f, sc.trade2_order1, v) || !parse_int(trim(v), t2o1) || (t2o1 != 1 && t2o1 != 2)) {
    err = "bad Trade2_Order1"; return false;
  }
  out.is_trade = (t2o1 == 2);

  if (!field_str(f, sc.transact_time, v) || !parse_time_hhmmssmmm(v, out.time)) { err = "bad TransactTime"; return false; }
  if (!field_str(f, sc.price, v) || !parse_price_milli(v, out.price)) { err = "bad Price"; return false; }
  if (!field_str(f, sc.volume, v) || !parse_int(trim(v), out.volume) || out.volume < 0) { err = "bad Volume"; return false; }

  if (out.is_trade) {
    if (field_str(f, sc.trd_bs, v)) {
      v = trim(v);
      out.trd_bs = v.empty() ? '-' : v[0];
      if (out.trd_bs == 'B') out.side = Side::Buy;
      else if (out.trd_bs == 'S') out.side = Side::Sell;
      else out.side = Side::None;
    }
    opt_int(f, sc.trd_money, out.trd_money);
    opt_int(f, sc.trd_buy_no, out.trd_buy_no);
    opt_int(f, sc.trd_sell_no, out.trd_sell_no);
  } else {
    std::int64_t side = 0;
    if (field_str(f, sc.ord_side, v) && parse_int(trim(v), side)) {
      out.side = (side == 1) ? Side::Buy : (side == 2) ? Side::Sell : Side::None;
    }
    if (field_str(f, sc.ord_type, v)) {
      v = trim(v);
      out.ord_type = v.empty() ? 0 : v[0];
    }
  }
  opt_int(f, sc.ord_no, out.ord_no);
  opt_int(f, sc.biz_index, out.biz_index);
  return true;
}

bool parse_snapshot(const SnapshotSchema& sc, std::string_view line, Snapshot& out, std::string& err) {
  std::vector<std::string_view> f;
  split_csv(line, f);
  out = Snapshot{};

  std::string_view v;
  if (!field_str(f, sc.instrument, v)) { err = "missing InstrumentID"; return false; }
  v = trim(v);
  if (v.empty() || v.size() > 12) { err = "bad InstrumentID"; return false; }
  out.instrument = make_symbol(v.data(), v.size());

  if (!field_str(f, sc.update_time, v) || !parse_time_hhmmssmmm(v, out.time)) { err = "bad UpdateTime"; return false; }
  if (!field_str(f, sc.last, v) || !parse_price_milli(v, out.last)) { err = "bad LastPrice"; return false; }

  opt_int(f, sc.seq, out.seq);
  opt_price(f, sc.pre_close, out.pre_close);
  opt_price(f, sc.open_px, out.open_px);
  opt_price(f, sc.high, out.high);
  opt_price(f, sc.low, out.low);
  opt_int(f, sc.trade_volume, out.cum_trade_volume);
  opt_int(f, sc.total_bid_vol, out.total_bid_vol);
  opt_int(f, sc.total_ask_vol, out.total_ask_vol);
  if (sc.iopv >= 0) {
    PriceI iopv = 0;
    opt_price(f, sc.iopv, iopv);
    out.iopv = iopv;
    out.iopv_valid = iopv > 0;
  }

  for (int k = 0; k < 10; ++k) {
    if (sc.bid_px[k] < 0 && sc.ask_px[k] < 0) break;
    PriceI p = 0; QtyI q = 0; std::int64_t n = 0;
    opt_price(f, sc.bid_px[k], p);
    opt_int(f, sc.bid_vol[k], q);
    opt_int(f, sc.bid_orders[k], n);
    out.bids[k] = BookLevel{p, q, static_cast<std::int32_t>(n)};
    p = 0; q = 0; n = 0;
    opt_price(f, sc.ask_px[k], p);
    opt_int(f, sc.ask_vol[k], q);
    opt_int(f, sc.ask_orders[k], n);
    out.asks[k] = BookLevel{p, q, static_cast<std::int32_t>(n)};
  }
  return true;
}

bool sse_order_is_cancel(const TickEvent& t) {
  // SSE vendor schema for this project annotates only OrdType A=limit, S=other
  // (see docs/data_dictionary.md); no reliable explicit cancel marker, so SSE
  // cancel decode is treated as UNRELIABLE and returns false (engine emits NaN
  // for cancel-gated factors and logs a warning). If the data dictionary later
  // confirms a cancel marker (e.g. 'D'), wire it here and flip the reliability.
  (void)t;
  return false;
}

bool szse_order_is_cancel(const TickEvent& t) {
  // SZSE L2 order stream carries cancel orders with OrderType 'X' (per SZSE
  // interface spec; confirmed mapping in docs/data_dictionary.md).
  return t.ord_type == 'X';
}

bool sse_cancel_decode_reliable() { return false; }
bool szse_cancel_decode_reliable() { return true; }

bool cancel_decode_reliable(const std::string& exchange) {
  return exchange == "szse" ? szse_cancel_decode_reliable() : sse_cancel_decode_reliable();
}

bool order_is_cancel(const TickEvent& t, const std::string& exchange) {
  return exchange == "szse" ? szse_order_is_cancel(t) : sse_order_is_cancel(t);
}

bool is_etf_code(const Symbol& s, const std::string& exchange) {
  if (s.size < 2) return false;
  auto p2 = [&](const char* p) { return s.data[0] == p[0] && s.data[1] == p[1]; };
  if (exchange == "sse") {
    return p2("50") || p2("51") || p2("52") || p2("56") || p2("58");
  }
  if (exchange == "szse") {
    return p2("15") || p2("16");
  }
  return false;
}

}  // namespace hftaf
