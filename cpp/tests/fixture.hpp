// fixture.hpp — deterministic synthetic SSE L2 fixture, written as .csv.gz at
// test time via zlib (no checked-in binary blobs, no Python dependency).
//
// Contents (exchange sse, date 20250603, one channel):
//   * 510300 (ETF): 61 snapshots every 3s from 09:30:00.000 to 09:33:00.000,
//     5 levels/side; between snapshots: 2 deep adds, 2 trades (one per side),
//     1 best-bid add. Prices wobble deterministically so factors move.
//   * 510500 (ETF): two snapshots (k=8 two-sided, k=9 one-sided + no IOPV) to
//     exercise FLAG_ONE_SIDED_BOOK | FLAG_IOPV_INVALID.
//   * 600000 (stock): one snapshot + one order tick; must be filtered out.
//   * One tick exactly at snapshot time t_5 (merge boundary: tick.time <= U).
// variant 0: clean SeqNo stream. variant 1: one SeqNo gap (skips 47..49).
#pragma once
#include <cstdio>
#include <cstdint>
#include <sstream>
#include <string>
#include <zlib.h>

namespace hftaft {

struct FixturePaths { std::string tick_gz; std::string snap_gz; };

inline std::string fmt_time(std::int64_t ms) {
  const std::int64_t hh = ms / 3600000;
  const std::int64_t mm = (ms / 60000) % 60;
  const std::int64_t ss = (ms / 1000) % 60;
  const std::int64_t mmm = ms % 1000;
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%02lld%02lld%02lld%03lld",
                (long long)hh, (long long)mm, (long long)ss, (long long)mmm);
  return std::string(buf);
}

inline std::string fmt_price(std::int64_t milli) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%lld.%03lld",
                (long long)(milli / 1000), (long long)(milli % 1000));
  return std::string(buf);
}

inline FixturePaths write_fixture(const std::string& dir, int variant) {
  const std::int64_t t0 = (9 * 3600 + 30 * 60) * 1000LL;   // 09:30:00.000
  const int n_snap = 61;                                    // every 3s to 09:33:00

  // ---------------- snapshot file ----------------
  std::ostringstream sn;
  sn << "InstrumentID,UpdateTime,LastPrice,PreClosePrice,OpenPrice,HighPrice,LowPrice,"
        "TradeVolume,IOPV,TotalBidVolume,TotalAskVolume";
  for (int k = 0; k < 5; ++k) sn << ",BidPrice" << k << ",BidVolume" << k;
  for (int k = 0; k < 5; ++k) sn << ",AskPrice" << k << ",AskVolume" << k;
  sn << "\n";

  auto snap_row = [&](const char* inst, std::int64_t t, std::int64_t bid1, std::int64_t ask1,
                      std::int64_t last, bool with_iopv, bool two_sided) {
    sn << inst << "," << fmt_time(t) << "," << fmt_price(last) << ","
       << fmt_price(3990) << "," << fmt_price(3995) << "," << fmt_price(last + 3) << ","
       << fmt_price(last - 3) << "," << 100000 << ",";
    if (with_iopv) sn << fmt_price(4000);
    std::int64_t tb = 0, ta = 0;
    for (int k = 0; k < 5; ++k) { tb += 1000 + 100 * k; ta += 1000 + 100 * k; }
    sn << "," << tb << "," << ta;
    for (int k = 0; k < 5; ++k) sn << "," << fmt_price(bid1 - 2 * k) << "," << (1000 + 100 * k);
    for (int k = 0; k < 5; ++k) {
      if (two_sided) sn << "," << fmt_price(ask1 + 2 * k) << "," << (1000 + 100 * k);
      else sn << ",,";                                     // empty => price 0 => one-sided
    }
    sn << "\n";
  };

  for (int k = 0; k < n_snap; ++k) {
    const std::int64_t t = t0 + 3000LL * k;
    const std::int64_t bid1 = 3998 + (k % 7);               // deterministic wobble
    const std::int64_t ask1 = bid1 + 4;                     // 4-tick spread
    snap_row("510300", t, bid1, ask1, bid1 + 2, true, true);
    if (k == 8) snap_row("510500", t, 2500, 2504, 2502, true, true);
    if (k == 9) snap_row("510500", t, 2500, 2504, 2502, false, false);
    if (k == 10) snap_row("600000", t, 1100, 1102, 1101, true, true);
  }

  // ---------------- tick file ----------------
  std::ostringstream tk;
  tk << "SeqNo,InstrumentID,Trade2_Order1,TransactTime,Price,Volume,OrdSide,OrdType,"
        "TrdBSFlag,TrdMoney,OrdNo,BizIndex\n";

  std::int64_t seq = 0;
  auto next_seq = [&]() {
    ++seq;
    if (variant == 1 && seq == 47) seq = 50;                // inject one gap
    return seq;
  };
  auto order_row = [&](const char* inst, std::int64_t t, std::int64_t px, std::int64_t vol,
                       int side, char ord_type) {
    tk << next_seq() << "," << inst << ",1," << fmt_time(t) << "," << fmt_price(px) << ","
       << vol << "," << side << "," << ord_type << ",,,," << "\n";
  };
  auto trade_row = [&](const char* inst, std::int64_t t, std::int64_t px, std::int64_t vol,
                       char bs) {
    tk << next_seq() << "," << inst << ",2," << fmt_time(t) << "," << fmt_price(px) << ","
       << vol << ",,," << bs << "," << (px * vol) << ",,\n";
  };

  for (int k = 0; k + 1 < n_snap; ++k) {
    const std::int64_t t = t0 + 3000LL * k;
    const std::int64_t bid1 = 3998 + (k % 7);
    const std::int64_t ask1 = bid1 + 4;
    order_row("510300", t + 500, bid1 - 10, 100, 1, 'A');   // deep bid add
    order_row("510300", t + 1000, ask1 + 10, 100, 2, 'A');  // deep ask add
    trade_row("510300", t + 1500, ask1, 100, 'B');          // buyer lifts best ask
    trade_row("510300", t + 2000, bid1, 100, 'S');          // seller hits best bid
    order_row("510300", t + 2500, bid1, 50, 1, 'A');        // add at best bid
    if (k == 5) order_row("510300", t + 3000, ask1 + 12, 10, 2, 'A');  // == snapshot time
    if (k == 10) order_row("600000", t + 700, 1100, 100, 1, 'A');      // filtered stock
  }

  FixturePaths paths;
  paths.tick_gz = dir + "/ticks.csv.gz";
  paths.snap_gz = dir + "/snapshots.csv.gz";

  gzFile gz = gzopen(paths.tick_gz.c_str(), "wb");
  const std::string tks = tk.str();
  gzwrite(gz, tks.data(), static_cast<unsigned>(tks.size()));
  gzclose(gz);

  gz = gzopen(paths.snap_gz.c_str(), "wb");
  const std::string sns = sn.str();
  gzwrite(gz, sns.data(), static_cast<unsigned>(sns.size()));
  gzclose(gz);
  return paths;
}

}  // namespace hftaft
