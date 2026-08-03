// hftaf/session.cpp — session constants and horizon-fit logic.
#include "hftaf/session.hpp"

namespace hftaf {

namespace {
constexpr TsMs t(int h, int m, int s = 0) {
  return (static_cast<TsMs>(h) * 3600 + static_cast<TsMs>(m) * 60 + s) * 1000;
}
}  // namespace

Session session_for(const std::string& exchange) {
  Session s;
  if (exchange == "szse") {
    // SZSE: continuous 09:30-11:30 / 13:00-14:57, then no-cancel closing
    // auction 14:57-15:00 (excluded from factors and labels).
    s.am_open = t(9, 30);
    s.am_close = t(11, 30);
    s.pm_open = t(13, 0);
    s.pm_close = t(14, 57);
    s.close_auction_start = t(14, 57);
  } else {
    // SSE funds: continuous 09:30-11:30 / 13:00-15:00, NO closing auction
    // (close = VWAP of last minute).
    s.am_open = t(9, 30);
    s.am_close = t(11, 30);
    s.pm_open = t(13, 0);
    s.pm_close = t(15, 0);
    s.close_auction_start = t(15, 0);   // == pm_close: venue has no auction
  }
  return s;
}

bool in_continuous_session(const Session& s, TsMs t) {
  return (t >= s.am_open && t < s.am_close) ||
         (t >= s.pm_open && t < s.close_auction_start);
}

namespace {
// 0 = AM block, 1 = PM block, -1 = outside session.
int block_of(const Session& s, TsMs t) {
  if (t >= s.am_open && t < s.am_close) return 0;
  if (t >= s.pm_open && t < s.close_auction_start) return 1;
  return -1;
}
}  // namespace

bool horizon_fits_session(const Session& s, TsMs t, TsMs horizon_ms) {
  const int b0 = block_of(s, t);
  const int b1 = block_of(s, t + horizon_ms);
  // Both endpoints must lie in the SAME continuous block; this rejects
  // windows crossing lunch, the close auction, or the session end.
  return b0 >= 0 && b0 == b1;
}

}  // namespace hftaf
