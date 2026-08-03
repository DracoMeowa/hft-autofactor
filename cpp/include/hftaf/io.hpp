// hftaf/io.hpp — streaming gzip line reader over zlib.
#pragma once
#include <cstdint>
#include <string>

namespace hftaf {

// Streaming gzip reader over zlib. Chunked; never holds the whole file.
class GzLineReader {
 public:
  explicit GzLineReader(const std::string& path, std::size_t chunk_bytes = 1 << 20);
  GzLineReader(const GzLineReader&) = delete;
  GzLineReader& operator=(const GzLineReader&) = delete;
  ~GzLineReader();

  bool ok() const;
  bool next_line(std::string& line);        // strips \r?\n; false on EOF/error
  std::uint64_t bytes_read() const;         // decompressed bytes consumed
  const std::string& error() const;

 private:
  bool refill();                            // read one chunk into buffer_; false on EOF/error

  void* gz_ = nullptr;                      // gzFile (void* to keep zlib out of this header)
  std::string buf_;                         // raw chunk bytes
  std::size_t pos_ = 0;                     // consume cursor in buf_
  std::size_t chunk_bytes_ = 1 << 20;
  std::uint64_t bytes_read_ = 0;
  bool eof_ = false;
  std::string error_;
  std::string path_;
};

}  // namespace hftaf
