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
  // buf_ starts EMPTY. refill() allocates and fills it on the first
  // next_line(). (Pre-sizing here would inject chunk_bytes_ of '\0' into the
  // first line, corrupting the CSV header.)
}

GzLineReader::~GzLineReader() {
  if (gz_) gzclose(static_cast<gzFile>(gz_));
}

bool GzLineReader::ok() const { return gz_ != nullptr && error_.empty(); }
std::uint64_t GzLineReader::bytes_read() const { return bytes_read_; }
const std::string& GzLineReader::error() const { return error_; }

bool GzLineReader::refill() {
  if (eof_ || !gz_) return false;
  buf_.assign(chunk_bytes_, '\0');  // allocate space for gzread
  int n = gzread(static_cast<gzFile>(gz_), buf_.data(),
                 static_cast<unsigned>(chunk_bytes_));
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
  buf_.resize(static_cast<std::size_t>(n));
  pos_ = 0;
  return true;
}

bool GzLineReader::next_line(std::string& line) {
  line.clear();
  if (!ok()) return false;
  for (;;) {
    // Refill whenever the buffer is exhausted (also covers the empty start).
    if (pos_ >= buf_.size()) {
      if (!refill()) {
        if (!error_.empty()) return false;
        return !line.empty();  // final unterminated line at EOF
      }
    }
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
    // Exhausted buffer without newline: keep partial line, loop to refill.
    line.append(buf_.data() + start, buf_.size() - start);
  }
}

}  // namespace hftaf
