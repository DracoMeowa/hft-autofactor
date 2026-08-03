// hftaf/output.cpp — fixed-schema CSV writer. Prices %.6f, NaN => empty cell,
// LF endings, UTF-8, header first.
#include "hftaf/output.hpp"
#include <cmath>
#include <cinttypes>
#include <string>

namespace hftaf {

OutputWriter::OutputWriter(const std::string& path, const std::vector<std::string>& factor_names,
                           const std::vector<int>& horizons_s) {
  fp_ = std::fopen(path.c_str(), "wb");
  if (!fp_) {
    error_ = "cannot open for write: " + path;
    return;
  }
  std::string header =
      "date,exchange,instrument,ts_ms,snap_seq,flags,"
      "mid_px,last_px,bid1_px,ask1_px,bid1_qty,ask1_qty,depth_bid5,depth_ask5";
  for (const auto& n : factor_names) { header += ','; header += n; }
  for (int h : horizons_s) { header += ",fwd_mid_ret_" + std::to_string(h) + "s"; }
  for (int h : horizons_s) { header += ",fwd_last_ret_" + std::to_string(h) + "s"; }
  header += '\n';
  if (std::fwrite(header.data(), 1, header.size(), fp_) != header.size()) {
    error_ = "write failed for header of " + path;
  }
  header_written_ = true;
}

OutputWriter::~OutputWriter() {
  if (fp_) std::fclose(fp_);
}

bool OutputWriter::ok() const { return fp_ != nullptr && error_.empty(); }
const std::string& OutputWriter::error() const { return error_; }

void OutputWriter::write_header() {
  // Header is emitted by the constructor; retained for interface compatibility.
}

void OutputWriter::write_double(double v) {
  char buf[64];
  if (std::isnan(v)) {
    buf[0] = ',';
    std::fwrite(buf, 1, 1, fp_);
    return;
  }
  int n = std::snprintf(buf, sizeof(buf), ",%.6f", v);
  std::fwrite(buf, 1, static_cast<std::size_t>(n), fp_);
}

void OutputWriter::write_row(const Row& r) {
  if (!ok()) return;
  char line[512];
  int n = std::snprintf(line, sizeof(line), "%s,%s,%s,%" PRId64 ",%" PRId64 ",%u",
                        r.date.c_str(), r.exchange.c_str(), symbol_to_string(r.instrument).c_str(),
                        static_cast<std::int64_t>(r.time),
                        static_cast<std::int64_t>(r.snap_seq),
                        static_cast<unsigned>(r.flags));
  std::fwrite(line, 1, static_cast<std::size_t>(n), fp_);

  write_double(r.mid_px);
  write_double(r.last_px);
  write_double(r.bid1_px);
  write_double(r.ask1_px);

  n = std::snprintf(line, sizeof(line), ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64,
                    static_cast<std::int64_t>(r.bid1_qty), static_cast<std::int64_t>(r.ask1_qty),
                    static_cast<std::int64_t>(r.depth_bid5), static_cast<std::int64_t>(r.depth_ask5));
  std::fwrite(line, 1, static_cast<std::size_t>(n), fp_);

  for (double v : r.factors) write_double(v);
  for (double v : r.fwd_mid) write_double(v);
  for (double v : r.fwd_last) write_double(v);
  std::fputc('\n', fp_);
}

void OutputWriter::finish() {
  if (fp_) {
    std::fflush(fp_);
    std::fclose(fp_);
    fp_ = nullptr;
  }
}

}  // namespace hftaf
