// hftaf-engine CLI driver.
//
// Usage:
//   hftaf-engine --exchange sse --date 20250603 --channel 3 \
//     --ticks /data/sse/202506/csv_XXXX/1_channel_3.csv.gz \
//     --snapshots /data/sse/202506/csv_XXXX/1_snapshot.csv.gz \
//     --out /data/factor_lzt/raw/20250603/sse_ch3.csv \
//     [--factors ofi_60s,oir,...] [--horizons 15,30,60,300,900] \
//     [--canaries] [--build-id SHA]
//
// Exit codes: 0 success; 2 usage/argument error; otherwise engine failure code.
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "hftaf/engine.hpp"

namespace {

void usage(std::ostream& os) {
  os << "usage: hftaf-engine --exchange {sse|szse} --date YYYYMMDD --channel N\n"
     << "                    --ticks <1_channel_N.csv.gz> --snapshots <1_snapshot.csv.gz>\n"
     << "                    --out <raw/.../{ex}_ch{N}.csv>\n"
     << "                    [--factors a,b,c] [--horizons 15,30,60,300,900]\n"
     << "                    [--canaries] [--build-id SHA]\n";
}

bool needs_value(int i, int argc) { return i + 1 < argc; }

std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(std::atoi(item.c_str()));
  }
  return out;
}

std::vector<std::string> parse_str_list(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    out.push_back(item);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  hftaf::JobPaths paths;
  hftaf::EngineOptions opts;
  bool have_exchange = false, have_date = false, have_channel = false;
  bool have_ticks = false, have_snap = false, have_out = false;
  bool horizons_set = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { usage(std::cout); return 0; }
    else if (a == "--exchange" && needs_value(i, argc)) { opts.exchange = argv[++i]; have_exchange = true; }
    else if (a == "--date" && needs_value(i, argc)) { opts.date = argv[++i]; have_date = true; }
    else if (a == "--channel" && needs_value(i, argc)) { opts.channel = std::atoi(argv[++i]); have_channel = true; }
    else if (a == "--ticks" && needs_value(i, argc)) { paths.tick_gz = argv[++i]; have_ticks = true; }
    else if (a == "--snapshots" && needs_value(i, argc)) { paths.snapshot_gz = argv[++i]; have_snap = true; }
    else if (a == "--out" && needs_value(i, argc)) { paths.out_csv = argv[++i]; have_out = true; }
    else if (a == "--factors" && needs_value(i, argc)) { opts.factors = parse_str_list(argv[++i]); }
    else if (a == "--horizons" && needs_value(i, argc)) { opts.horizons_s = parse_int_list(argv[++i]); horizons_set = true; }
    else if (a == "--canaries") { opts.include_canaries = true; }
    else if (a == "--build-id" && needs_value(i, argc)) { opts.build_id = argv[++i]; }
    else {
      std::cerr << "error: unknown or incomplete argument: " << a << "\n";
      usage(std::cerr);
      return 2;
    }
  }

  if (!have_exchange || !have_date || !have_channel || !have_ticks || !have_snap || !have_out) {
    std::cerr << "error: missing required argument(s)\n";
    usage(std::cerr);
    return 2;
  }
  if (opts.exchange != "sse" && opts.exchange != "szse") {
    std::cerr << "error: --exchange must be sse or szse\n";
    return 2;
  }
  if (opts.date.size() != 8) {
    std::cerr << "error: --date must be YYYYMMDD\n";
    return 2;
  }
  if (!horizons_set) opts.horizons_s = {15, 30, 60, 300, 900};
  if (opts.horizons_s.empty()) {
    std::cerr << "error: --horizons parsed to an empty list\n";
    return 2;
  }

  return hftaf::run_job(paths, opts, std::cerr);
}
