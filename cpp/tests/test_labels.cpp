// test_labels.cpp — strictly future-only labels: resolution at first snapshot
// >= t+H, ABSENT (NaN) across lunch/auction/session-end, no padding, day-end
// flush, invalid base prices.
#include <cmath>
#include <limits>
#include <vector>

#include "hftaf/labels.hpp"
#include "hftaf/session.hpp"
#include "test_util.hpp"

using namespace hftaf;

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

TsMs t(int h, int m, int s, int ms = 0) {
  return ((h * 60LL + m) * 60LL + s) * 1000LL + ms;
}

Snapshot mk(const Symbol& inst, TsMs time, PriceI bid, PriceI ask, PriceI last) {
  Snapshot s;
  s.time = time;
  s.instrument = inst;
  s.bids[0] = BookLevel{bid, 1000, 1};
  s.asks[0] = BookLevel{ask, 1000, 1};
  s.last = last;
  return s;
}

Row mk_row(const Snapshot& s, double mid, double last) {
  Row r;
  r.date = "20250603";
  r.exchange = "sse";
  r.instrument = s.instrument;
  r.time = s.time;
  r.mid_px = mid;
  r.last_px = last;
  return r;
}

}  // namespace

// Basic resolution: label = (P(t+H) - P(t)) / P(t) at the FIRST snapshot >= t+H.
static void test_basic_resolution() {
  const Session sess = session_for("sse");
  LabelBuilder lb(LabelConfig{{15, 60}}, sess);
  std::vector<Row> out;
  lb.set_sink([&](Row&& r) { out.push_back(std::move(r)); });

  const Symbol inst = make_symbol("510300", 6);
  const int N = 26;
  for (int k = 0; k < N; ++k) {
    const TsMs time = t(9, 30, 0) + 3000LL * k;
    const PriceI bid = 4000 + k;         // mid drifts up 1 milli per snapshot
    const Snapshot s = mk(inst, time, bid, bid + 4, bid + 2);
    lb.push(mk_row(s, (bid + 2) / 1000.0, (bid + 2) / 1000.0), s);
  }

  // Row 0 fully resolves once the snapshot at t+60s (k=20) has been pushed.
  CHECK(!out.empty());
  CHECK_EQ(out[0].time, t(9, 30, 0));
  CHECK_NEAR(out[0].fwd_mid[0], 5.0 / 4002.0, 1e-12);     // 15s: k=5
  CHECK_NEAR(out[0].fwd_mid[1], 20.0 / 4002.0, 1e-12);    // 60s: k=20
  CHECK_NEAR(out[0].fwd_last[0], 5.0 / 4002.0, 1e-12);
  CHECK_NEAR(out[0].fwd_last[1], 20.0 / 4002.0, 1e-12);

  // Rows 0..5 are fully resolved by the time k=25 is pushed.
  CHECK_EQ((int)out.size(), 6);
  for (std::size_t i = 1; i < out.size(); ++i)
    CHECK(out[i - 1].time < out[i].time);                 // ascending emission

  // Day-end flush emits the rest with ABSENT (NaN) unresolved labels.
  lb.end_instrument_day(inst);
  CHECK_EQ((int)out.size(), N);
  for (int i = 6; i < N; ++i) {
    CHECK(std::isnan(out[i].fwd_mid[0]));
    CHECK(std::isnan(out[i].fwd_mid[1]));
  }
  // A second flush is a no-op.
  lb.end_instrument_day(inst);
  CHECK_EQ((int)out.size(), N);
}

// Windows crossing lunch or leaving the session are ABSENT even if later
// snapshots exist (in another block or beyond the close).
static void test_session_edges() {
  const Session sess = session_for("sse");
  LabelBuilder lb(LabelConfig{{15}}, sess);
  std::vector<Row> out;
  lb.set_sink([&](Row&& r) { out.push_back(std::move(r)); });
  const Symbol inst = make_symbol("510300", 6);

  // 11:29:50 + 15s crosses into the lunch break => ABSENT.
  Snapshot s0 = mk(inst, t(11, 29, 50), 4000, 4004, 4002);
  lb.push(mk_row(s0, 4.002, 4.002), s0);
  Snapshot s1 = mk(inst, t(11, 29, 53), 4001, 4005, 4003);   // triggers resolution
  lb.push(mk_row(s1, 4.003, 4.003), s1);
  CHECK_EQ((int)out.size(), 1);
  CHECK(std::isnan(out[0].fwd_mid[0]));
  CHECK(std::isnan(out[0].fwd_last[0]));

  // 11:29:40 + 15s = 11:29:55 fits inside the AM block => present.
  LabelBuilder lb2(LabelConfig{{15}}, sess);
  std::vector<Row> out2;
  lb2.set_sink([&](Row&& r) { out2.push_back(std::move(r)); });
  Snapshot a0 = mk(inst, t(11, 29, 40), 4000, 4004, 4002);
  lb2.push(mk_row(a0, 4.002, 4.002), a0);
  Snapshot a1 = mk(inst, t(11, 29, 55), 4010, 4014, 4012);
  lb2.push(mk_row(a1, 4.012, 4.012), a1);
  CHECK_EQ((int)out2.size(), 1);
  CHECK_NEAR(out2[0].fwd_mid[0], 10.0 / 4002.0, 1e-12);

  // SSE has no closing auction, but the close at 15:00 still bounds labels:
  // 14:59:40 + 15s fits; 14:59:50 + 20s runs past the close. SZSE auction
  // already excludes 14:56:50 + 15s.
  CHECK(horizon_fits_session(session_for("sse"), t(14, 59, 40), 15000));
  CHECK(!horizon_fits_session(session_for("szse"), t(14, 56, 50), 15000));
  CHECK(horizon_fits_session(session_for("szse"), t(14, 56, 40), 15000));
  CHECK(!horizon_fits_session(session_for("sse"), t(14, 59, 50), 20000));  // past close
  CHECK(!horizon_fits_session(session_for("sse"), t(11, 29, 50), 15000));  // lunch
  // Close boundary is exclusive: a print exactly at 15:00 is outside.
  CHECK(!in_continuous_session(session_for("sse"), t(15, 0, 0)));
  CHECK(in_continuous_session(session_for("sse"), t(14, 59, 59, 999)));
  CHECK(!in_continuous_session(session_for("szse"), t(14, 57, 0)));
}

// SZSE: window reaching into the closing auction is ABSENT even with a
// snapshot inside the auction window available.
static void test_szse_auction() {
  const Session sess = session_for("szse");
  LabelBuilder lb(LabelConfig{{15}}, sess);
  std::vector<Row> out;
  lb.set_sink([&](Row&& r) { out.push_back(std::move(r)); });
  const Symbol inst = make_symbol("159915", 6);

  Snapshot s0 = mk(inst, t(14, 56, 50), 2500, 2504, 2502);
  Row r0 = mk_row(s0, 2.502, 2.502);
  r0.exchange = "szse";
  lb.push(std::move(r0), s0);
  Snapshot s1 = mk(inst, t(14, 57, 30), 2510, 2514, 2512);   // inside auction
  Row r1 = mk_row(s1, 2.512, 2.512);
  r1.exchange = "szse";
  lb.push(std::move(r1), s1);
  CHECK_EQ((int)out.size(), 1);
  CHECK(std::isnan(out[0].fwd_mid[0]));                      // ABSENT, not padded
}

// Invalid base price at t => ABSENT label for that series only.
static void test_invalid_base() {
  const Session sess = session_for("sse");
  LabelBuilder lb(LabelConfig{{15}}, sess);
  std::vector<Row> out;
  lb.set_sink([&](Row&& r) { out.push_back(std::move(r)); });
  const Symbol inst = make_symbol("510300", 6);

  // One-sided book at t: mid NaN, last valid.
  Snapshot s0 = mk(inst, t(9, 30, 0), 4000, 0, 4002);
  lb.push(mk_row(s0, kNan, 4.002), s0);
  Snapshot s1 = mk(inst, t(9, 30, 15), 4004, 4008, 4006);
  lb.push(mk_row(s1, 4.006, 4.006), s1);
  CHECK_EQ((int)out.size(), 1);
  CHECK(std::isnan(out[0].fwd_mid[0]));
  CHECK_NEAR(out[0].fwd_last[0], 4.0 / 4002.0, 1e-12);
}

int main() {
  test_basic_resolution();
  test_session_edges();
  test_szse_auction();
  test_invalid_base();
  return hftaft::finish("test_labels");
}
