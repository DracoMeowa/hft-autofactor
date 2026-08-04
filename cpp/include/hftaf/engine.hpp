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

  // --- replay cache -------------------------------------------------------
  // Cache-build mode (build_cache_dir non-empty): stream the inputs once with
  // the exact raw-mode merge/SeqNo semantics and write a replay cache:
  //   <dir>/events.csv.gz — verbatim target tick rows ("T,<line>") and target
  //     snapshot rows ("S,<gap_bit>,<line>") in original interleaved order,
  //   <dir>/meta.json     — exchange/date/channel, original header lines,
  //     raw input sizes, event counts, instrument list.
  // No factor rows are computed and --out is ignored. cache_instruments picks
  // the targets; empty => every ETF in the channel.
  std::string build_cache_dir;
  std::vector<std::string> cache_instruments;
  // Replay mode (use_cache_dir non-empty): compute factor rows from a cache
  // written by cache-build. --ticks/--snapshots are ignored; --exchange/
  // --date/--channel must match the cache meta. Output is byte-identical to
  // running the raw inputs with the same options, restricted to the cached
  // instruments (gap flags replayed from the recorded per-snapshot bits).
  std::string use_cache_dir;
};

// Streams both inputs; merge rule: ticks ordered by SeqNo, snapshots by UpdateTime, all ticks
// with TransactTime <= U processed before snapshot U. Emits rows ordered (instrument asc,
// time asc). Writes <out_csv>.tmp then atomically renames; writes <out_csv>.meta.json
// (input sizes, build_id, row count, factor list). Returns 0 on success.
int run_job(const JobPaths& paths, const EngineOptions& opts, std::ostream& log);

}  // namespace hftaf
