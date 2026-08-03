// hftaf/io.cpp — streaming gzip line reader over zlib (chunked gzread).
#include "hftaf/io.hpp"
#include <zlib.h>

namespace hftaf {

GzLineReader::GzLineReader(const std::string& path, std::size_t chunk_bytes)
    : chunk_bytes_(chunk_bytes < 4096 ? 4096 : chunk_bytes), path_(path) {
  gz_ = gzopen(path.c_str(), "rb");
  if (!gz_) {
    error_ = "cannot open " + path;
    return;
  }
  buf_.resize(chunk_bytes_);
}

GzLineReader::~GzLineReader() {
  if (gz_) gzclose(static_cast<gzFile>(gz_));
}

bool GzLineReader::ok() const { return gz_ != nullptr && error_.empty(); }
std::uint64_t GzLineReader::bytes_read() const { return bytes_read_; }
const std::string& GzLineReader::error() const { return error_; }

bool GzLineReader::refill() {
  if (eof_ || !gz_) return false;
  int n = gzread(static_cast<gzFile>(gz_), &buf_[0], static_cast<unsigned>(buf_.size()));
  if (n < 0) {
    int errnum = 0;
    const char* msg = gzerror(static_cast<gzFile>(gz_), &errnum);
    error_ = std::string("gzread error in ") + path_ + ": " + (msg ? msg : "?");
    return false;
  }
  if (n == 0) {
    eof_ = true;
    return false;
  }
  bytes_read_ += static_cast<std::uint64_t>(n);
  pos_ = 0;
  buf_.resize(static_cast<std::size_t>(n));
  return true;
}

bool GzLineReader::next_line(std::string& line) {
  line.clear();
  if (!ok()) return false;
  for (;;) {
    // Consume from the current buffer until we hit '\n' or run dry.
    std::size_t start = pos_;
    while (pos_ < buf_.size() && buf_[pos_] != '\n') ++pos_;
    if (pos_ < buf_.size()) {
      // Found newline; copy up to it, strip trailing \r.
      std::size_t end = pos_;
      ++pos_;  // consume '\n'
      if (end > start && buf_[end - 1] == '\r') --end;
      line.append(buf_.data() + start, end - start);
      return true;
    }
    // Exhausted buffer without newline: keep what we have, refill.
    line.append(buf_.data() + start, buf_.size() - start);
    if (!refill()) {
      // EOF or error: emit final unterminated line if non-empty.
      if (!error_.empty()) return false;
      return !line.empty();
    }
  }
}

}  // namespace hftaf
