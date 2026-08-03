// hftaf/session.hpp — continuous-session calendars per exchange.
//
// Venue asymmetry (primary source: SSE trading rules 2023 rev. section 2.4.2):
//   SSE funds trade continuously 09:30-11:30 and 13:00-15:00 with NO closing
//   auction (close = VWAP of last minute). SZSE securities (incl. SZSE ETFs)
//   trade continuously 09:30-11:30 / 13:00-14:57, then a no-cancel closing
//   auction 14:57-15:00 which is EXCLUDED from factors and labels.
// Call auctions (09:15-09:25) are never part of factor rows.
#pragma once
#include <string>
#include "hftaf/types.hpp"

namespace hftaf {

struct Session {
  TsMs am_open, am_close, pm_open, pm_close;   // continuous-trading bounds
  TsMs close_auction_start;                    // == pm_close when venue has none
};

// SSE funds: 09:30-11:30, 13:00-15:00 continuous, NO closing auction.
// SZSE:      09:30-11:30, 13:00-14:57 continuous, auction 14:57-15:00 excluded.
Session session_for(const std::string& exchange);

// True iff t is in [am_open, am_close) or [pm_open, close_auction_start).
// Close boundaries are exclusive: a print exactly at 11:30:00.000 / 15:00:00.000
// (or 14:57:00.000 on SZSE) is NOT in the continuous session.
bool in_continuous_session(const Session& s, TsMs t);

// False when t or t+horizon_ms leaves the continuous session or the window
// spans the lunch break / close auction / session end. Labels use this to emit
// ABSENT (empty cell) instead of padding across session edges.
bool horizon_fits_session(const Session& s, TsMs t, TsMs horizon_ms);

}  // namespace hftaf
