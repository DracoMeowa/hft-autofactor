// test_decode.cpp — CSV split, exact parsers, schema resolution by name with
// aliases, tick/snapshot decode, cancel classification, ETF filter.
#include <string>
#include <vector>

#include "hftaf/decode.hpp"
#include "test_util.hpp"

using namespace hftaf;

static std::vector<std::string_view> hdr(std::initializer_list<const char*> cols) {
  // Views must stay alive for the caller; leak-free for a unit test.
  static std::vector<std::vector<std::string>> pool;
  pool.emplace_back();
  auto& v = pool.back();
  v.reserve(cols.size());   // no realloc => string_views below stay valid
  std::vector<std::string_view> out;
  for (auto c : cols) { v.emplace_back(c); out.push_back(v.back()); }
  return out;
}

static void test_split_csv() {
  std::vector<std::string_view> f;
  CHECK_EQ(split_csv("a,b,c", f), (std::size_t)3);
  CHECK(f[0] == "a" && f[1] == "b" && f[2] == "c");
  CHECK_EQ(split_csv("a,,c", f), (std::size_t)3);
  CHECK(f[1].empty());
  CHECK_EQ(split_csv("a,", f), (std::size_t)2);   // trailing comma => empty last field
  CHECK_EQ(split_csv("", f), (std::size_t)1);     // single empty field
  CHECK_EQ(split_csv("only", f), (std::size_t)1);
}

static void test_parse_price_milli() {
  PriceI p = -1;
  CHECK(parse_price_milli("3.456", p) && p == 3456);
  CHECK(parse_price_milli("3.45", p) && p == 3450);   // padded, not 3.45 -> float
  CHECK(parse_price_milli("3", p) && p == 3000);
  CHECK(parse_price_milli("0.001", p) && p == 1);
  CHECK(parse_price_milli("3.4567", p) && p == 3456); // >3 frac digits truncated
  CHECK(parse_price_milli("123456.789", p) && p == 123456789LL);
  CHECK(parse_price_milli(" 3.5 ", p) && p == 3500);  // whitespace trimmed
  CHECK(!parse_price_milli("", p));
  CHECK(!parse_price_milli("-3.5", p));               // signed rejected
  CHECK(!parse_price_milli("abc", p));
  CHECK(!parse_price_milli("1..2", p));
  CHECK(!parse_price_milli(".", p));
  CHECK(!parse_price_milli("3.5a", p));
}

static void test_parse_int() {
  std::int64_t v = 0;
  CHECK(parse_int("123", v) && v == 123);
  CHECK(parse_int("-5", v) && v == -5);
  CHECK(parse_int("+7", v) && v == 7);
  CHECK(parse_int("0", v) && v == 0);
  CHECK(!parse_int("", v));
  CHECK(!parse_int("1a", v));
  CHECK(!parse_int("-", v));
}

static void test_parse_time() {
  TsMs t = -1;
  CHECK(parse_time_hhmmssmmm("093000000", t) && t == 34200000LL);
  CHECK(parse_time_hhmmssmmm("093000", t) && t == 34200000LL);   // seconds only
  CHECK(parse_time_hhmmssmmm("145959999", t) &&
        t == ((14 * 60 + 59) * 60 + 59) * 1000LL + 999);
  // Real SSE/SZSE dumps write HHMMSSmmm as an integer, dropping leading
  // zeros: accept and right-justify positionally.
  CHECK(parse_time_hhmmssmmm("93000000", t) && t == 34200000LL);
  CHECK(parse_time_hhmmssmmm("91400650", t) &&
        t == ((9 * 60 + 14) * 60 + 0) * 1000LL + 650);
  CHECK(parse_time_hhmmssmmm("60000900", t) && t == 6 * 3600000LL + 900);
  CHECK(parse_time_hhmmssmmm("930000", t) && t == 34200000LL);
  CHECK(!parse_time_hhmmssmmm("246060000", t));   // hh=24 invalid
  CHECK(!parse_time_hhmmssmmm("096000000", t));   // mm=60 invalid
  CHECK(!parse_time_hhmmssmmm("09300000a", t));
  CHECK(!parse_time_hhmmssmmm("1234567890", t));  // >9 digits
  CHECK(!parse_time_hhmmssmmm("", t));
}

static void test_tick_schema() {
  std::string err;
  TickSchema sc;
  // Canonical SSE-style header.
  auto h1 = hdr({"SeqNo", "InstrumentID", "Trade2_Order1", "TransactTime", "Price",
                 "Volume", "OrdSide", "OrdType", "TrdBSFlag", "TrdMoney", "OrdNo"});
  CHECK(make_tick_schema(h1, sc, err));
  CHECK_EQ(sc.seq, 0);
  CHECK_EQ(sc.instrument, 1);
  CHECK_EQ(sc.trade2_order1, 2);
  CHECK_EQ(sc.transact_time, 3);
  CHECK_EQ(sc.price, 4);
  CHECK_EQ(sc.volume, 5);
  CHECK_EQ(sc.ord_side, 6);
  CHECK_EQ(sc.ord_type, 7);
  CHECK_EQ(sc.trd_bs, 8);
  CHECK_EQ(sc.trd_money, 9);
  CHECK_EQ(sc.ord_no, 10);
  CHECK_EQ(sc.biz_index, -1);

  // Alias + reordered header resolves identically by NAME.
  auto h2 = hdr({"ApplSeqNum", "SecCode", "TradeOrderFlag", "TransTime", "Price",
                 "Qty", "OrderSide", "OrderType", "BSFlag"});
  CHECK(make_tick_schema(h2, sc, err));
  CHECK_EQ(sc.seq, 0);
  CHECK_EQ(sc.instrument, 1);
  CHECK_EQ(sc.volume, 5);
  CHECK_EQ(sc.trd_bs, 8);

  // Case-insensitive match.
  auto h3 = hdr({"seqno", "INSTRUMENTID", "Trade2_Order1", "TransactTime", "price",
                 "Volume", "OrdSide", "OrdType", "TrdBSFlag"});
  CHECK(make_tick_schema(h3, sc, err));

  // Missing required column => failure naming it.
  auto h4 = hdr({"SeqNo", "InstrumentID", "Trade2_Order1", "TransactTime",
                 "Volume", "OrdSide", "OrdType", "TrdBSFlag"});
  CHECK(!make_tick_schema(h4, sc, err));
  CHECK(err.find("Price") != std::string::npos);
}

static void test_snapshot_schema() {
  std::string err;
  SnapshotSchema sc;
  auto h = hdr({"InstrumentID", "UpdateTime", "LastPrice", "SeqNo", "IOPV",
                "BidPrice0", "BidVolume0", "AskPrice0", "AskVolume0",
                "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"});
  CHECK(make_snapshot_schema(h, sc, err));
  CHECK_EQ(sc.instrument, 0);
  CHECK_EQ(sc.update_time, 1);
  CHECK_EQ(sc.last, 2);
  CHECK_EQ(sc.seq, 3);
  CHECK_EQ(sc.iopv, 4);
  CHECK_EQ(sc.bid_px[0], 5);
  CHECK_EQ(sc.ask_vol[0], 8);
  CHECK_EQ(sc.bid_px[1], 9);
  CHECK_EQ(sc.ask_px[1], 11);
  CHECK_EQ(sc.bid_px[2], -1);   // contiguous levels stop at missing column

  // Real SSE/SZSE dumps use bracketed level names (BidPrice[0], ...).
  auto hb = hdr({"InstrumentID", "UpdateTime", "LastPrice", "IOPV",
                 "BidPrice[0]", "BidVolume[0]", "AskPrice[0]", "AskVolume[0]",
                 "BidPrice[1]", "BidVolume[1]", "AskPrice[1]", "AskVolume[1]"});
  CHECK(make_snapshot_schema(hb, sc, err));
  CHECK_EQ(sc.bid_px[0], 4);
  CHECK_EQ(sc.ask_vol[0], 7);
  CHECK_EQ(sc.bid_px[1], 8);
  CHECK_EQ(sc.ask_vol[1], 11);

  // Missing best ask level => failure.
  auto h2 = hdr({"InstrumentID", "UpdateTime", "LastPrice",
                 "BidPrice0", "BidVolume0", "AskPrice0"});
  CHECK(!make_snapshot_schema(h2, sc, err));
  CHECK(err.find("AskVolume0") != std::string::npos);
}

static void test_parse_tick_rows() {
  std::string err;
  TickSchema sc;
  auto h = hdr({"SeqNo", "InstrumentID", "Trade2_Order1", "TransactTime", "Price",
                "Volume", "OrdSide", "OrdType", "TrdBSFlag", "TrdMoney"});
  CHECK(make_tick_schema(h, sc, err));

  TickEvent t;
  // Order row: buy limit add.
  CHECK(parse_tick(sc, "17,510300,1,093001500,3.998,100,1,A,,", t, err));
  CHECK_EQ(t.seq, 17);
  CHECK(symbol_to_string(t.instrument) == "510300");
  CHECK(!t.is_trade);
  CHECK_EQ(t.time, 34201500LL);
  CHECK_EQ(t.price, 3998);
  CHECK_EQ(t.volume, 100);
  CHECK(t.side == Side::Buy);
  CHECK_EQ((int)t.ord_type, (int)'A');

  // Trade row: buyer aggressor.
  CHECK(parse_tick(sc, "18,510300,2,093002000,4.002,200,,,B,800400", t, err));
  CHECK(t.is_trade);
  CHECK_EQ(t.trd_bs, (int)'B');
  CHECK(t.side == Side::Buy);
  CHECK_EQ(t.trd_money, 800400);

  // '-' print has no aggressor.
  CHECK(parse_tick(sc, "19,510300,2,093002500,4.000,50,,,-,0", t, err));
  CHECK(t.is_trade);
  CHECK_EQ((int)t.trd_bs, (int)'-');
  CHECK(t.side == Side::None);

  // Malformed rows rejected.
  CHECK(!parse_tick(sc, "xx,510300,1,093001500,3.998,100,1,A,,", t, err));  // bad seq
  CHECK(!parse_tick(sc, "17,510300,3,093001500,3.998,100,1,A,,", t, err));  // bad flag
  CHECK(!parse_tick(sc, "17,510300,1,09300150,3.998,100,1,A,,", t, err));   // 8-digit time
  CHECK(!parse_tick(sc, "17,510300,1,093001500,3.9x8,100,1,A,,", t, err));  // bad price
}

static void test_parse_snapshot_rows() {
  std::string err;
  SnapshotSchema sc;
  auto h = hdr({"InstrumentID", "UpdateTime", "LastPrice", "IOPV",
                "BidPrice0", "BidVolume0", "AskPrice0", "AskVolume0",
                "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"});
  CHECK(make_snapshot_schema(h, sc, err));

  Snapshot s;
  CHECK(parse_snapshot(sc, "510300,093000000,4.000,3.995,3.998,2000,4.002,1000,3.996,1500,4.004,1200", s, err));
  CHECK(symbol_to_string(s.instrument) == "510300");
  CHECK_EQ(s.time, 34200000LL);
  CHECK_EQ(s.last, 4000);
  CHECK(s.iopv_valid);
  CHECK_EQ(s.iopv, 3995);
  CHECK_EQ(s.bids[0].price, 3998);
  CHECK_EQ(s.bids[0].volume, 2000);
  CHECK_EQ(s.asks[0].price, 4002);
  CHECK_EQ(s.asks[1].volume, 1200);
  CHECK_EQ(s.seq, -1);           // file lacks seq column => stays -1

  // Empty IOPV => invalid; empty ask => price 0 (one-sided).
  CHECK(parse_snapshot(sc, "510500,093027000,2.502,,2.500,900,,,2.498,700,,", s, err));
  CHECK(!s.iopv_valid);
  CHECK_EQ(s.asks[0].price, 0);
}

static void test_cancel_and_universe() {
  TickEvent t;
  t.ord_type = 'X';
  CHECK(szse_order_is_cancel(t));
  CHECK(!sse_order_is_cancel(t));
  t.ord_type = 'A';
  CHECK(!szse_order_is_cancel(t));

  CHECK(szse_cancel_decode_reliable());
  CHECK(!sse_cancel_decode_reliable());
  CHECK(cancel_decode_reliable("szse"));
  CHECK(!cancel_decode_reliable("sse"));

  t.ord_type = 'X';
  CHECK(order_is_cancel(t, "szse"));
  CHECK(!order_is_cancel(t, "sse"));

  CHECK(is_etf_code(make_symbol("510300", 6), "sse"));
  CHECK(is_etf_code(make_symbol("588000", 6), "sse"));
  CHECK(!is_etf_code(make_symbol("600000", 6), "sse"));
  CHECK(is_etf_code(make_symbol("159915", 6), "szse"));
  CHECK(is_etf_code(make_symbol("161725", 6), "szse"));
  CHECK(!is_etf_code(make_symbol("000001", 6), "szse"));
  CHECK(!is_etf_code(make_symbol("159915", 6), "sse"));   // exchange matters
}

int main() {
  test_split_csv();
  test_parse_price_milli();
  test_parse_int();
  test_parse_time();
  test_tick_schema();
  test_snapshot_schema();
  test_parse_tick_rows();
  test_parse_snapshot_rows();
  test_cancel_and_universe();
  return hftaft::finish("test_decode");
}
