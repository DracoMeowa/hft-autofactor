// hftaf/output.hpp — fixed-schema CSV writer for the interchange files.
#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include "hftaf/types.hpp"

namespace hftaf {

// CSV column order (see docs/interchange_format.md):
// date,exchange,instrument,ts_ms,snap_seq,flags,mid_px,last_px,bid1_px,ask1_px,
// bid1_qty,ask1_qty,depth_bid5,depth_ask5,<factor columns...>,
// fwd_mid_ret_15s,...,fwd_mid_ret_900s,fwd_last_ret_15s,...,fwd_last_ret_900s
//
// Formatting rules:
//   * prices and factor values printed with %.6f (CNY / dimensionless)
//   * NaN => empty cell (never the literal "nan")
//   * LF line endings, UTF-8, header row first
//   * quantities/seq/ts_ms/flags printed as integers
class OutputWriter {
 public:
  OutputWriter(const std::string& path, const std::vector<std::string>& factor_names,
               const std::vector<int>& horizons_s);
  OutputWriter(const OutputWriter&) = delete;
  OutputWriter& operator=(const OutputWriter&) = delete;
  ~OutputWriter();

  bool ok() const;
  void write_header();
  void write_row(const Row& r);
  void finish();   // flush + close
  const std::string& error() const;

 private:
  void write_double(double v);    // %.6f or empty for NaN
  FILE* fp_ = nullptr;
  std::string error_;
  bool header_written_ = false;
};

}  // namespace hftaf
