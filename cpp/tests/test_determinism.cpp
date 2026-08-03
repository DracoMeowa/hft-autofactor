// test_determinism.cpp — end-to-end engine runs over the synthetic gz fixture:
//   * byte-identical output across repeated runs (determinism contract)
//   * exact CSV schema (column order), sorted rows, universe filtering
//   * warm-up gating, ABSENT forward labels, one-sided/IOPV flags
//   * SeqNo-gap flagging on the injected-gap variant
//   * canary delayed-attach actually reads the future (mask test must fail)
//   * meta.json sidecar contents, atomic write (no .tmp left behind)
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hftaf/engine.hpp"
#include "hftaf/factors.hpp"
#include "hftaf/types.hpp"
#include "fixture.hpp"
#include "test_util.hpp"

namespace fs = std::filesystem;
using namespace hftaf;

namespace {

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Table {
  std::vector<std::string> header;
  std::vector<std::vector<std::string>> rows;
  int col(const std::string& name) const {
    for (std::size_t i = 0; i < header.size(); ++i)
      if (header[i] == name) return static_cast<int>(i);
    return -1;
  }
};

Table parse_csv(const std::string& text) {
  Table t;
  auto lines = hftaft::split_lines(text);
  if (lines.empty()) return t;
  t.header = hftaft::split_fields(lines[0]);
  for (std::size_t i = 1; i < lines.size(); ++i) t.rows.push_back(hftaft::split_fields(lines[i]));
  return t;
}

EngineOptions base_opts() {
  EngineOptions o;
  o.exchange = "sse";
  o.date = "20250603";
  o.channel = 3;
  o.horizons_s = {15, 30, 60, 300, 900};
  o.build_id = "test-build";
  return o;
}

constexpr std::int64_t kT0 = 34200000LL;   // 09:30:00.000

}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() / "hftaf_test_determinism";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  // ---------------- variant 0: clean stream ----------------
  const fs::path v0 = dir / "v0";
  fs::create_directories(v0, ec);
  auto fx = hftaft::write_fixture(v0.string(), 0);

  JobPaths p1{fx.tick_gz, fx.snap_gz, (v0 / "out1.csv").string()};
  std::ostringstream log1;
  CHECK_EQ(run_job(p1, base_opts(), log1), 0);

  JobPaths p2{fx.tick_gz, fx.snap_gz, (v0 / "out2.csv").string()};
  std::ostringstream log2;
  CHECK_EQ(run_job(p2, base_opts(), log2), 0);

  const std::string csv1 = read_file(p1.out_csv);
  const std::string csv2 = read_file(p2.out_csv);
  CHECK(!csv1.empty());
  CHECK(csv1 == csv2);                                    // determinism contract
  CHECK(!fs::exists(std::string(p1.out_csv) + ".tmp"));   // atomic rename done

  Table t = parse_csv(csv1);

  // Exact schema: fixed columns, default factor order, horizon columns.
  std::string expected_header =
      "date,exchange,instrument,ts_ms,snap_seq,flags,"
      "mid_px,last_px,bid1_px,ask1_px,bid1_qty,ask1_qty,depth_bid5,depth_ask5";
  for (const auto& n : kDefaultFactorNames) expected_header += "," + n;
  for (int h : {15, 30, 60, 300, 900}) expected_header += ",fwd_mid_ret_" + std::to_string(h) + "s";
  for (int h : {15, 30, 60, 300, 900}) expected_header += ",fwd_last_ret_" + std::to_string(h) + "s";
  {
    std::ostringstream hdr;
    for (std::size_t i = 0; i < t.header.size(); ++i) {
      if (i) hdr << ",";
      hdr << t.header[i];
    }
    CHECK(hdr.str() == expected_header);
  }

  const int c_inst = t.col("instrument"), c_ts = t.col("ts_ms"), c_flags = t.col("flags");
  const int c_seq = t.col("snap_seq");
  CHECK(c_inst >= 0 && c_ts >= 0 && c_flags >= 0);
  CHECK(!t.rows.empty());

  // Ordering + universe: sorted (instrument asc, ts asc); no stock rows.
  for (std::size_t i = 1; i < t.rows.size(); ++i) {
    const auto& a = t.rows[i - 1];
    const auto& b = t.rows[i];
    CHECK(a[c_inst] < b[c_inst] ||
          (a[c_inst] == b[c_inst] && std::stoll(a[c_ts]) < std::stoll(b[c_ts])));
  }
  for (const auto& r : t.rows) CHECK(r[c_inst] != "600000");

  auto find_row = [&](const std::string& inst, std::int64_t ts) -> const std::vector<std::string>* {
    for (const auto& r : t.rows)
      if (r[c_inst] == inst && std::stoll(r[c_ts]) == ts) return &r;
    return nullptr;
  };

  // Fixture lacks a snapshot SeqNo column => -1 propagated.
  if (const auto* r = find_row("510300", kT0)) CHECK_EQ(std::stoll((*r)[c_seq]), -1LL);

  // Row 0 (09:30:00): snapshot factors ready, tick factors + rv_60s warming up.
  const int c_oir = t.col("oir"), c_ofi = t.col("ofi_60s"), c_rv60 = t.col("rv_60s");
  const int c_spread = t.col("quoted_spread_ticks");
  const int c_fm15 = t.col("fwd_mid_ret_15s"), c_fm900 = t.col("fwd_mid_ret_900s");
  const int c_fm300 = t.col("fwd_mid_ret_300s"), c_fl15 = t.col("fwd_last_ret_15s");
  if (const auto* r = find_row("510300", kT0)) {
    CHECK(!(*r)[c_oir].empty());
    CHECK(!(*r)[c_spread].empty());
    CHECK((*r)[c_ofi].empty());                          // warm-up: ABSENT, not 0
    CHECK((*r)[c_rv60].empty());
    CHECK(!(*r)[c_fm15].empty());                        // resolved at 09:30:15
    CHECK(!(*r)[c_fl15].empty());
    CHECK((*r)[c_fm300].empty());                        // no snapshot 300s later
    CHECK((*r)[c_fm900].empty());                        // ABSENT, never padded
  } else {
    CHECK(false);
  }

  // rv_60s warm-up boundary: empty at 09:30:57, present at 09:31:00.
  if (const auto* r = find_row("510300", kT0 + 57000)) CHECK((*r)[c_rv60].empty());
  else CHECK(false);
  if (const auto* r = find_row("510300", kT0 + 60000)) CHECK(!(*r)[c_rv60].empty());
  else CHECK(false);

  // OFI warm-up boundary: first tick at 09:30:00.5 => first valid at 09:31:03.
  if (const auto* r = find_row("510300", kT0 + 60000)) CHECK((*r)[c_ofi].empty());
  else CHECK(false);
  if (const auto* r = find_row("510300", kT0 + 63000)) CHECK(!(*r)[c_ofi].empty());
  else CHECK(false);

  // 510500 k=9 snapshot is one-sided and lacks IOPV.
  const int c_flag_bits = c_flags;
  if (const auto* r = find_row("510500", kT0 + 27000)) {
    const unsigned long fl = std::stoul((*r)[c_flag_bits]);
    CHECK((fl & FLAG_ONE_SIDED_BOOK) != 0);
    CHECK((fl & FLAG_IOPV_INVALID) != 0);
  } else {
    CHECK(false);
  }

  // Clean variant: no seq-gap flags anywhere.
  for (const auto& r : t.rows)
    CHECK((std::stoul(r[c_flags]) & FLAG_SEQ_GAP_BEFORE) == 0);

  // meta.json sidecar.
  const std::string meta = read_file(std::string(p1.out_csv) + ".meta.json");
  CHECK(meta.find("\"build_id\": \"test-build\"") != std::string::npos);
  CHECK(meta.find("\"exchange\": \"sse\"") != std::string::npos);
  CHECK(meta.find("\"rows\": " + std::to_string(t.rows.size())) != std::string::npos);
  CHECK(meta.find("\"quoted_spread_ticks\"") != std::string::npos);

  // ---------------- canaries: delayed attach must see the future ----------------
  JobPaths p3{fx.tick_gz, fx.snap_gz, (v0 / "out3.csv").string()};
  EngineOptions copts = base_opts();
  copts.include_canaries = true;
  std::ostringstream log3;
  CHECK_EQ(run_job(p3, copts, log3), 0);
  Table tc = parse_csv(read_file(p3.out_csv));
  const int c_fm = tc.col("future_mid_15s"), c_ft = tc.col("future_trade_sign");
  CHECK(c_fm >= 0 && c_ft >= 0);
  int nonempty_fm = 0, nonempty_ft = 0;
  for (const auto& r : tc.rows) {
    if (r[c_inst] == "510300") {
      if (!r[c_fm].empty()) ++nonempty_fm;
      if (!r[c_ft].empty()) ++nonempty_ft;
    }
  }
  CHECK(nonempty_fm > 0);   // canary got future data => py-eval mask test must fail
  CHECK(nonempty_ft > 0);

  // Canary run is deterministic too.
  JobPaths p4{fx.tick_gz, fx.snap_gz, (v0 / "out4.csv").string()};
  std::ostringstream log4;
  CHECK_EQ(run_job(p4, copts, log4), 0);
  CHECK(read_file(p3.out_csv) == read_file(p4.out_csv));

  // ---------------- variant 1: injected SeqNo gap ----------------
  const fs::path v1 = dir / "v1";
  fs::create_directories(v1, ec);
  auto fxg = hftaft::write_fixture(v1.string(), 1);
  JobPaths pg{fxg.tick_gz, fxg.snap_gz, (v1 / "out_gap.csv").string()};
  std::ostringstream logg;
  CHECK_EQ(run_job(pg, base_opts(), logg), 0);
  Table tg = parse_csv(read_file(pg.out_csv));
  int gap_rows = 0;
  bool gap_at_expected = false;
  for (const auto& r : tg.rows) {
    if ((std::stoul(r[tg.col("flags")]) & FLAG_SEQ_GAP_BEFORE) != 0) {
      ++gap_rows;
      if (r[tg.col("instrument")] == "510300" && std::stoll(r[tg.col("ts_ms")]) == kT0 + 30000)
        gap_at_expected = true;
    }
  }
  CHECK_EQ(gap_rows, 1);           // exactly the next snapshot row of the ETF
  CHECK(gap_at_expected);

  // ---------------- argument errors ----------------
  {
    EngineOptions bad = base_opts();
    bad.exchange = "xyz";
    std::ostringstream lg;
    CHECK_EQ(run_job(p1, bad, lg), 2);
  }
  {
    EngineOptions bad = base_opts();
    bad.factors = {"no_such_factor"};
    std::ostringstream lg;
    CHECK_EQ(run_job(p1, bad, lg), 2);
  }
  {
    JobPaths missing{"/nonexistent/ticks.csv.gz", "/nonexistent/snaps.csv.gz",
                     (v0 / "outx.csv").string()};
    std::ostringstream lg;
    CHECK_EQ(run_job(missing, base_opts(), lg), 2);
  }

  fs::remove_all(dir, ec);
  return hftaft::finish("test_determinism");
}
