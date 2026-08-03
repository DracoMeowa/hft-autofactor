// test_util.hpp — minimal dependency-free check harness for hftaf unit tests.
#pragma once
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace hftaft {

inline int& failures() { static int n = 0; return n; }
inline int& checks() { static int n = 0; return n; }

inline void report(bool ok, const char* file, int line, const std::string& what) {
  ++checks();
  if (!ok) {
    ++failures();
    std::printf("FAIL %s:%d: %s\n", file, line, what.c_str());
  }
}

template <typename A, typename B>
void report_eq(const A& a, const B& b, const char* file, int line, const char* expr) {
  ++checks();
  if (!(a == b)) {
    ++failures();
    std::ostringstream os;
    os << expr << "  (lhs=" << a << " rhs=" << b << ")";
    std::printf("FAIL %s:%d: %s\n", file, line, os.str().c_str());
  }
}

inline int finish(const char* name) {
  std::printf("%s: %d checks, %d failures\n", name, checks(), failures());
  return failures() == 0 ? 0 : 1;
}

// CSV helpers reused across tests.
inline std::vector<std::string> split_fields(const std::string& line) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      out.push_back(line.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

inline std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || text[i] == '\n') {
      if (i > start) out.push_back(text.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

}  // namespace hftaft

#define CHECK(cond) ::hftaft::report((cond), __FILE__, __LINE__, #cond)
#define CHECK_EQ(a, b) ::hftaft::report_eq((a), (b), __FILE__, __LINE__, #a " == " #b)
#define CHECK_NEAR(a, b, eps) \
  ::hftaft::report(std::fabs((a) - (b)) <= (eps), __FILE__, __LINE__, #a " ~= " #b)
