// test_cache.cpp — replay-cache round trip (engine optimization ②):
//   * build a cache from the raw fixture, replay it, and require the output
//     CSV AND meta sidecar to be byte-identical to the raw run
//   * explicit-target cache: replay output == raw output restricted to target
//   * snapshot-before-first-tick edge (510500): rows survive targeted caching
//   * SeqNo-gap variant: gap flag bit survives the cache round trip
//   * canary mode works over replay (same future-peeking output as raw)
//   * argument guards: date/channel mismatch, missing cache, build+use clash
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

EngineOptions base_opts() {
  EngineOptions o;
  o.exchange = "sse";
  o.date = "20250603";
  o.channel = 3;
  o.horizons_s = {15, 30, 60, 300, 900};
  o.build_id = "test-build";
  return o;
}

// raw CSV restricted to rows whose `instrument` column equals `inst`.
std::string filter_rows(const std::string& csv, const std::string& inst) {
  auto lines = hftaft::split_lines(csv);
  if (lines.empty()) return "";
  auto header = hftaft::split_fields(lines[0]);
  int c_inst = -1;
  for (std::size_t i = 0; i < header.size(); ++i)
    if (header[i] == "instrument") c_inst = static_cast<int>(i);
  std::string out = lines[0] + "\n";
  for (std::size_t i = 1; i < lines.size(); ++i) {
    auto f = hftaft::split_fields(lines[i]);
    if (c_inst >= 0 && c_inst < static_cast<int>(f.size()) && f[c_inst] == inst)
      out += lines[i] + "\n";
  }
  return out;
}

}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() / "hftaf_test_cache";
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);

  // ================= variant 0: clean stream =================
  const fs::path v0 = dir / "v0";
  fs::create_directories(v0, ec);
  auto fx = hftaft::write_fixture(v0.string(), 0);
  std::ostringstream lg;

  // raw baseline
  JobPaths praw{fx.tick_gz, fx.snap_gz, (v0 / "raw.csv").string()};
  CHECK_EQ(run_job(praw, base_opts(), lg), 0);
  const std::string raw_csv = read_file(praw.out_csv);
  const std::string raw_meta = read_file(std::string(praw.out_csv) + ".meta.json");
  CHECK(!raw_csv.empty());

  // --- build: dynamic (all ETFs) ---
  EngineOptions bo = base_opts();
  bo.build_cache_dir = (v0 / "cache_all").string();
  JobPaths pbuild{fx.tick_gz, fx.snap_gz, (v0 / "never_written.csv").string()};
  CHECK_EQ(run_job(pbuild, bo, lg), 0);
  CHECK(fs::exists(v0 / "cache_all" / "events.csv.gz"));
  CHECK(fs::exists(v0 / "cache_all" / "meta.json"));
  CHECK(!fs::exists(v0 / "cache_all" / "events.csv.gz.tmp"));
  CHECK(!fs::exists(v0 / "never_written.csv"));          // build ignores --out

  // --- replay: byte-identical CSV and meta sidecar ---
  EngineOptions ro = base_opts();
  ro.use_cache_dir = (v0 / "cache_all").string();
  JobPaths preplay{"", "", (v0 / "replay.csv").string()};
  CHECK_EQ(run_job(preplay, ro, lg), 0);
  CHECK(read_file(preplay.out_csv) == raw_csv);
  CHECK(read_file(std::string(preplay.out_csv) + ".meta.json") == raw_meta);

  // replay determinism (second run)
  JobPaths preplay2{"", "", (v0 / "replay2.csv").string()};
  CHECK_EQ(run_job(preplay2, ro, lg), 0);
  CHECK(read_file(preplay2.out_csv) == raw_csv);

  // --- build: explicit single target ---
  EngineOptions bo1 = base_opts();
  bo1.build_cache_dir = (v0 / "cache_510300").string();
  bo1.cache_instruments = {"510300"};
  CHECK_EQ(run_job(pbuild, bo1, lg), 0);
  EngineOptions ro1 = base_opts();
  ro1.use_cache_dir = (v0 / "cache_510300").string();
  JobPaths preplay3{"", "", (v0 / "replay_510300.csv").string()};
  CHECK_EQ(run_job(preplay3, ro1, lg), 0);
  CHECK(read_file(preplay3.out_csv) == filter_rows(raw_csv, "510300"));

  // --- snapshot-before-first-tick edge: 510500's first snapshot (k=8)
  //     precedes its only tick (k=11); targeted caching must keep its rows ---
  EngineOptions bo2 = base_opts();
  bo2.build_cache_dir = (v0 / "cache_510500").string();
  bo2.cache_instruments = {"510500"};
  CHECK_EQ(run_job(pbuild, bo2, lg), 0);
  EngineOptions ro2 = base_opts();
  ro2.use_cache_dir = (v0 / "cache_510500").string();
  JobPaths preplay4{"", "", (v0 / "replay_510500.csv").string()};
  CHECK_EQ(run_job(preplay4, ro2, lg), 0);
  const std::string only_510500 = filter_rows(raw_csv, "510500");
  CHECK(!only_510500.empty());
  CHECK(read_file(preplay4.out_csv) == only_510500);

  // --- canaries over replay == canaries over raw ---
  EngineOptions co = base_opts();
  co.include_canaries = true;
  JobPaths prawc{fx.tick_gz, fx.snap_gz, (v0 / "raw_canary.csv").string()};
  CHECK_EQ(run_job(prawc, co, lg), 0);
  EngineOptions roc = co;
  roc.use_cache_dir = (v0 / "cache_all").string();
  JobPaths preplayc{"", "", (v0 / "replay_canary.csv").string()};
  CHECK_EQ(run_job(preplayc, roc, lg), 0);
  CHECK(read_file(preplayc.out_csv) == read_file(prawc.out_csv));

  // --- argument guards ---
  {
    EngineOptions bad = base_opts();                      // date mismatch
    bad.use_cache_dir = (v0 / "cache_all").string();
    bad.date = "20250604";
    JobPaths p{"", "", (v0 / "x1.csv").string()};
    CHECK_EQ(run_job(p, bad, lg), 2);
  }
  {
    EngineOptions bad = base_opts();                      // channel mismatch
    bad.use_cache_dir = (v0 / "cache_all").string();
    bad.channel = 5;
    JobPaths p{"", "", (v0 / "x2.csv").string()};
    CHECK_EQ(run_job(p, bad, lg), 2);
  }
  {
    EngineOptions bad = base_opts();                      // missing cache dir
    bad.use_cache_dir = (v0 / "no_such_cache").string();
    JobPaths p{"", "", (v0 / "x3.csv").string()};
    CHECK_EQ(run_job(p, bad, lg), 2);
  }
  {
    EngineOptions bad = base_opts();                      // build+use clash
    bad.use_cache_dir = (v0 / "cache_all").string();
    bad.build_cache_dir = (v0 / "cache_x").string();
    JobPaths p{fx.tick_gz, fx.snap_gz, (v0 / "x4.csv").string()};
    CHECK_EQ(run_job(p, bad, lg), 2);
  }
  {
    EngineOptions bad = base_opts();                      // bad target name
    bad.build_cache_dir = (v0 / "cache_bad").string();
    bad.cache_instruments = {""};
    JobPaths p{fx.tick_gz, fx.snap_gz, (v0 / "x5.csv").string()};
    CHECK_EQ(run_job(p, bad, lg), 2);
  }

  // ================= variant 1: injected SeqNo gap =================
  const fs::path v1 = dir / "v1";
  fs::create_directories(v1, ec);
  auto fxg = hftaft::write_fixture(v1.string(), 1);

  JobPaths prawg{fxg.tick_gz, fxg.snap_gz, (v1 / "raw_gap.csv").string()};
  CHECK_EQ(run_job(prawg, base_opts(), lg), 0);
  const std::string raw_gap_csv = read_file(prawg.out_csv);

  EngineOptions bog = base_opts();
  bog.build_cache_dir = (v1 / "cache_gap").string();
  JobPaths pbuildg{fxg.tick_gz, fxg.snap_gz, (v1 / "unused.csv").string()};
  CHECK_EQ(run_job(pbuildg, bog, lg), 0);
  EngineOptions rog = base_opts();
  rog.use_cache_dir = (v1 / "cache_gap").string();
  JobPaths preplayg{"", "", (v1 / "replay_gap.csv").string()};
  CHECK_EQ(run_job(preplayg, rog, lg), 0);
  CHECK(read_file(preplayg.out_csv) == raw_gap_csv);

  // The gap flag must actually be present in the round-tripped output.
  {
    auto lines = hftaft::split_lines(raw_gap_csv);
    auto header = hftaft::split_fields(lines[0]);
    int c_flags = -1;
    for (std::size_t i = 0; i < header.size(); ++i)
      if (header[i] == "flags") c_flags = static_cast<int>(i);
    int gap_rows = 0;
    for (std::size_t i = 1; i < lines.size(); ++i) {
      auto f = hftaft::split_fields(lines[i]);
      if (c_flags >= 0 && (std::stoul(f[c_flags]) & FLAG_SEQ_GAP_BEFORE) != 0) ++gap_rows;
    }
    CHECK_EQ(gap_rows, 1);
  }

  fs::remove_all(dir, ec);
  return hftaft::finish("test_cache");
}
