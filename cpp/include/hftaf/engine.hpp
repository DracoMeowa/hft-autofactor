// hftaf/engine.hpp — per-job streaming pipeline driver.
#pragma once
#include <iosfwd>
#include <string>
#include <vector>

namespace hftaf {

struct JobPaths {
  std::string tick_gz;       // /data/{ex}/YYYYMM/csv_*/1_channel_N.csv.gz
  std::string snapshot_gz;   // /data/{ex}/YYYYMM/csv_*/1_snapshot.csv.gz
  std::string out_csv;       // /data/factor_lzt/raw/YYYYMMDD/{ex}_ch{N}.csv
};

struct EngineOptions {
  std::string exchange;                  // "sse" | "szse"
  std::string date;                      // YYYYMMDD
  int channel = 0;                       // metadata only
  std::vector<int> horizons_s = {15, 30, 60, 300, 900};
  std::vector<std::string> factors;      // empty => full default registry
  bool include_canaries = false;
  std::string build_id;                  // embedded in .meta.json sidecar
};

// Streams both inputs; merge rule: ticks ordered by SeqNo, snapshots by UpdateTime, all ticks
// with TransactTime <= U processed before snapshot U. Emits rows ordered (instrument asc,
// time asc). Writes <out_csv>.tmp then atomically renames; writes <out_csv>.meta.json
// (input sizes, build_id, row count, factor list). Returns 0 on success.
int run_job(const JobPaths& paths, const EngineOptions& opts, std::ostream& log);

}  // namespace hftaf
