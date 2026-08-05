// test_factors.cpp — in-memory registry checks: exact values for snapshot
// factors, 60s window warm-up, SZSE cancel-gated factors, OFI zero-flow case,
// trade-size/arrival wishlist columns (avg size, count, large-share, gap,
// cum volume), canary look-ahead semantics, causal value_at() refusal,
// registry errors.
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "hftaf/book.hpp"
#include "hftaf/factors.hpp"
#include "hftaf/session.hpp"
#include "test_util.hpp"

using namespace hftaf;

namespace {

TsMs t(int h, int m, int s, int ms = 0) {
  return ((h * 60LL + m) * 60LL + s) * 1000LL + ms;
}

Snapshot make_snap(const Symbol& inst, TsMs time) {
  Snapshot s;
  s.time = time;
  s.instrument = inst;
  s.last = 4000;
  // Constant book: spread 4 ticks, bid qty 2000 vs ask qty 1000.
  s.bids[0] = BookLevel{3998, 2000, 3};
  s.asks[0] = BookLevel{4002, 1000, 2};
  for (int k = 1; k < 5; ++k) {
    s.bids[k] = BookLevel{3998 - 2 * k, 500, 1};
    s.asks[k] = BookLevel{4002 + 2 * k, 500, 1};
  }
  return s;
}

TickEvent order(TsMs time, const Symbol& inst, Side side, PriceI price, QtyI vol,
                char type = 'A') {
  TickEvent e;
  e.time = time;
  e.instrument = inst;
  e.is_trade = false;
  e.side = side;
  e.price = price;
  e.volume = vol;
  e.ord_type = type;
  return e;
}

TickEvent trade(TsMs time, const Symbol& inst, QtyI vol, char bs) {
  TickEvent e;
  e.time = time;
  e.instrument = inst;
  e.is_trade = true;
  e.side = (bs == 'B') ? Side::Buy : (bs == 'S') ? Side::Sell : Side::None;
  e.price = 4000;
  e.volume = vol;
  e.trd_bs = bs;
  return e;
}

Snapshot make_snap_cum(const Symbol& inst, TsMs time, QtyI cum_vol) {
  Snapshot s = make_snap(inst, time);
  s.cum_trade_volume = cum_vol;
  return s;
}

IFactor* find(std::vector<std::unique_ptr<IFactor>>& reg, const std::string& name) {
  for (auto& f : reg)
    if (f && f->name() == name) return f.get();
  return nullptr;
}

}  // namespace

// Drive the full stream: 41 snapshots 3s apart (covers the 60s warm-up),
// each interval carrying 2 adds + 2 cancels (balanced, deterministic).
static void test_szse_window_factors() {
  const Symbol inst = make_symbol("159915", 6);
  const Session sess = session_for("szse");
  FactorContext ctx{"20250603", "szse", sess};

  // include_canaries=true: this test exercises future_mid_15s's look-ahead
  // value_at() semantics, and the canary guard refuses canary names in the
  // plain factors list. The implicit append then also adds future_trade_sign,
  // which the find()-based checks below simply ignore.
  auto reg = make_registry({"quoted_spread_ticks", "microprice_dev", "oir", "ofi_60s",
                            "order_arrival_60s", "cancel_ratio_60s", "future_mid_15s"},
                           true);
  for (auto& f : reg) {
    f->on_day_start(ctx);
    f->on_instrument_day_start(inst);
  }

  BookState book;
  const TsMs t0 = t(9, 30, 0);
  for (int k = 0; k <= 40; ++k) {
    const TsMs ts = t0 + 3000LL * k;
    if (k > 0) {
      // Interval (t_{k-1}, t_k): adds deep (no quote changes => OFI contrib 0),
      // then cancels; adds == cancels and buys == sells on counts.
      const TsMs base = t0 + 3000LL * (k - 1);
      TickEvent evs[4] = {
          order(base + 1000, inst, Side::Buy, 3988, 100, 'A'),
          order(base + 1500, inst, Side::Sell, 4012, 100, 'A'),
          order(base + 2000, inst, Side::Buy, 3988, 50, 'X'),    // SZSE cancel
          order(base + 2500, inst, Side::Sell, 4012, 50, 'X'),   // SZSE cancel
      };
      for (auto& e : evs) {
        book.apply_order(e);
        for (auto& f : reg) f->on_tick(e, book);
      }
    }
    const Snapshot s = make_snap(inst, ts);
    book.apply_snapshot(s);
    for (auto& f : reg) f->on_snapshot(s, book);
  }

  double v;
  // Snapshot-native factors: exact values on the constant book.
  CHECK(find(reg, "quoted_spread_ticks")->value(inst, v));
  CHECK_NEAR(v, 4.0, 1e-12);                                // 4 ticks of 1 milli
  CHECK(find(reg, "oir")->value(inst, v));
  CHECK_NEAR(v, 1000.0 / 3000.0, 1e-12);                    // (2000-1000)/3000
  CHECK(find(reg, "microprice_dev")->value(inst, v));
  // micro - mid = (ap-bp)(bq-aq)/(2(bq+aq)) = 4*1000/6000 milli => bps of 4000 milli
  CHECK_NEAR(v, (4.0 * 1000.0 / 6000.0) / 4000.0 * 1e4, 1e-9);

  // OFI: no best-quote changes across the stream => sum of contributions 0.
  CHECK(find(reg, "ofi_60s")->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);

  // Balanced adds => zero arrival imbalance; cancels are 50% of order messages.
  CHECK(find(reg, "order_arrival_60s")->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);
  CHECK(find(reg, "cancel_ratio_60s")->value(inst, v));
  CHECK_NEAR(v, 0.5, 1e-12);

  // Canary: value() stays false; value_at() reaches 15s into the future.
  IFactor* canary = find(reg, "future_mid_15s");
  CHECK(canary != nullptr);
  CHECK(canary->is_canary());
  CHECK(!canary->value(inst, v));
  CHECK(canary->value_at(inst, t0, v));                     // first snap >= 09:30:15
  CHECK_NEAR(v, 4.0, 1e-12);                                // mid of 3998/4002 book
  // Causal factors refuse point-in-time (future) queries.
  CHECK(!find(reg, "oir")->value_at(inst, t0, v));
}

// Warm-up contract: before a full 60s of history the tick factors emit NaN
// (value() == false), never a partial-window number.
static void test_warmup() {
  const Symbol inst = make_symbol("159915", 6);
  const Session sess = session_for("szse");
  FactorContext ctx{"20250603", "szse", sess};

  std::vector<std::unique_ptr<IFactor>> reg;
  reg.push_back(make_tick_factor("ofi_60s"));
  reg.push_back(make_tick_factor("trade_imbalance_60s"));
  reg.push_back(make_snapshot_factor("rv_60s"));
  CHECK(reg[0] && reg[1] && reg[2]);
  for (auto& f : reg) {
    f->on_day_start(ctx);
    f->on_instrument_day_start(inst);
  }

  BookState book;
  const TsMs t0 = t(9, 30, 0);
  double v;
  // Snapshots at 0, 30, 59s: all inside warm-up. Causal order: snapshot at
  // t0+off first, then one order tick in (t0+off, t0+off+3s).
  for (TsMs off : {0LL, 30000LL, 59000LL}) {
    const Snapshot s = make_snap(inst, t0 + off);
    book.apply_snapshot(s);
    for (auto& f : reg) f->on_snapshot(s, book);
    CHECK(!reg[0]->value(inst, v));   // ofi_60s warming up
    CHECK(!reg[2]->value(inst, v));   // rv_60s warming up
    TickEvent e = order(t0 + off + 1000, inst, Side::Buy, 3988, 100, 'A');
    book.apply_order(e);
    for (auto& f : reg) f->on_tick(e, book);
  }
  // After the 60s mark, keep feeding 3s snapshots until the window holds
  // >= 80% of the nominal return count (16 of 20) => valid.
  for (TsMs off = 60000LL; off <= 120000LL; off += 3000LL) {
    const Snapshot s = make_snap(inst, t0 + off);
    book.apply_snapshot(s);
    for (auto& f : reg) f->on_snapshot(s, book);
  }
  CHECK(reg[2]->value(inst, v));
  CHECK(std::isfinite(v) && v >= 0.0);
}

// SSE cancel decode is unreliable: cancel-gated factors must emit NaN there
// while still producing values on SZSE from the same event stream.
static void test_sse_cancel_gating() {
  const Symbol inst = make_symbol("510300", 6);
  const Session sess = session_for("sse");
  FactorContext ctx{"20250603", "sse", sess};

  auto reg = make_registry({"order_arrival_60s", "cancel_ratio_60s"}, false);
  for (auto& f : reg) {
    f->on_day_start(ctx);
    f->on_instrument_day_start(inst);
  }
  BookState book;
  const TsMs t0 = t(9, 30, 0);
  for (int k = 0; k <= 25; ++k) {
    const TsMs ts = t0 + 3000LL * k;
    TickEvent e = order(ts + 1000, inst, Side::Buy, 3988, 100, 'X');
    book.apply_order(e);
    for (auto& f : reg) f->on_tick(e, book);
    const Snapshot s = make_snap(inst, ts);
    book.apply_snapshot(s);
    for (auto& f : reg) f->on_snapshot(s, book);
  }
  double v;
  for (auto& f : reg) CHECK(!f->value(inst, v));   // NaN on SSE, never wrong values
}

// Wishlist trade-size / arrival columns: exact values on a hand-built trade
// stream, 60s warm-up, '-' print inclusion, window pruning, trade_gap_ms
// semantics, cum_trade_vol monotonicity guard.
static void test_trade_window_factors() {
  const Symbol inst = make_symbol("510300", 6);
  const Session sess = session_for("sse");
  FactorContext ctx{"20250603", "sse", sess};

  auto reg = make_registry({"avg_trade_size_60s", "n_trades_60s",
                            "large_trade_share_60s", "trade_gap_ms",
                            "cum_trade_vol"}, false);
  CHECK_EQ((int)reg.size(), 5);
  for (auto& f : reg) {
    f->on_day_start(ctx);
    f->on_instrument_day_start(inst);
  }

  IFactor* avg = find(reg, "avg_trade_size_60s");
  IFactor* ntr = find(reg, "n_trades_60s");
  IFactor* lts = find(reg, "large_trade_share_60s");
  IFactor* gap = find(reg, "trade_gap_ms");
  IFactor* cum = find(reg, "cum_trade_vol");
  CHECK(avg && ntr && lts && gap && cum);

  BookState book;
  const TsMs t0 = t(9, 30, 0);
  double v;

  // Snapshot before any trade: window factors + gap absent; cum volume 0 is
  // a legitimate value (no trades yet), not warm-up.
  Snapshot s0 = make_snap_cum(inst, t0, 0);
  book.apply_snapshot(s0);
  for (auto& f : reg) f->on_snapshot(s0, book);
  CHECK(!avg->value(inst, v));
  CHECK(!gap->value(inst, v));
  CHECK(cum->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);

  // Trade stream: five 100-unit prints every 10s (one '-' print included --
  // size/count statistics are side-blind), then one 1000-unit whale print.
  TickEvent trades[] = {
      trade(t0 + 1000, inst, 100, 'B'),
      trade(t0 + 11000, inst, 100, 'S'),
      trade(t0 + 21000, inst, 100, '-'),   // unattributed: still counted
      trade(t0 + 31000, inst, 100, 'B'),
      trade(t0 + 41000, inst, 100, 'S'),
      trade(t0 + 55000, inst, 1000, 'B'),
  };

  // Snapshot at t0+30s: only 29s since the first trade => warm-up.
  for (auto& e : trades) {
    if (e.time >= t0 + 30000) break;
    book.apply_trade(e);
    for (auto& f : reg) f->on_tick(e, book);
  }
  Snapshot s1 = make_snap_cum(inst, t0 + 30000, 300);
  book.apply_snapshot(s1);
  for (auto& f : reg) f->on_snapshot(s1, book);
  CHECK(!avg->value(inst, v));
  CHECK(!ntr->value(inst, v));
  CHECK(!lts->value(inst, v));
  CHECK(gap->value(inst, v));               // instantaneous: no 60s warm-up
  CHECK_NEAR(v, 30000.0 - 21000.0, 1e-9);   // 9s since the last print
  CHECK(cum->value(inst, v));
  CHECK_NEAR(v, 300.0, 1e-12);

  // Snapshot at t0+59s: 58s since first trade => still warm-up.
  for (auto& e : trades) {
    if (e.time >= t0 + 59000 || e.time < t0 + 30000) continue;
    book.apply_trade(e);
    for (auto& f : reg) f->on_tick(e, book);
  }
  Snapshot s2 = make_snap_cum(inst, t0 + 59000, 1500);
  book.apply_snapshot(s2);
  for (auto& f : reg) f->on_snapshot(s2, book);
  CHECK(!avg->value(inst, v));
  CHECK(gap->value(inst, v));
  CHECK_NEAR(v, 4000.0, 1e-9);              // whale print 4s ago

  // Snapshot at t0+62s: warm (61s since first print). The 60s window prunes
  // the t0+1s print (ts < 62s-60s): 4x100 + 1x1000 => n=5, total 1400.
  Snapshot s3 = make_snap_cum(inst, t0 + 62000, 1500);
  book.apply_snapshot(s3);
  for (auto& f : reg) f->on_snapshot(s3, book);
  CHECK(ntr->value(inst, v));
  CHECK_NEAR(v, 5.0, 1e-12);
  CHECK(avg->value(inst, v));
  CHECK_NEAR(v, 1400.0 / 5.0, 1e-12);       // 280 units/trade
  CHECK(lts->value(inst, v));
  // k_large = ceil(5/10) = 1 => the whale alone: 1000/1400.
  CHECK_NEAR(v, 1000.0 / 1400.0, 1e-12);
  CHECK(gap->value(inst, v));
  CHECK_NEAR(v, 7000.0, 1e-9);

  // Snapshot at t0+72s: the window prunes the t0+11s print too:
  // 3x100 + 1x1000 => n=4, total 1300, share 1000/1300.
  Snapshot s4 = make_snap_cum(inst, t0 + 72000, 1500);
  book.apply_snapshot(s4);
  for (auto& f : reg) f->on_snapshot(s4, book);
  CHECK(ntr->value(inst, v));
  CHECK_NEAR(v, 4.0, 1e-12);
  CHECK(avg->value(inst, v));
  CHECK_NEAR(v, 325.0, 1e-12);
  CHECK(lts->value(inst, v));
  CHECK_NEAR(v, 1000.0 / 1300.0, 1e-12);

  // cum_trade_vol monotonicity guard: an intra-day decrease emits NaN and
  // resumes once the series recovers above the pre-drop level.
  CHECK(cum->value(inst, v));
  CHECK_NEAR(v, 1500.0, 1e-12);
  Snapshot s5 = make_snap_cum(inst, t0 + 75000, 1200);   // feed anomaly
  book.apply_snapshot(s5);
  for (auto& f : reg) f->on_snapshot(s5, book);
  CHECK(!cum->value(inst, v));
  Snapshot s6 = make_snap_cum(inst, t0 + 76000, 1600);   // recovered
  book.apply_snapshot(s6);
  for (auto& f : reg) f->on_snapshot(s6, book);
  CHECK(cum->value(inst, v));
  CHECK_NEAR(v, 1600.0, 1e-12);
}

// Real-stream ordering guard for trade_gap_ms: SSE publishes different
// instruments' snapshots with different per-cycle UpdateTime phases, and the
// engine's shared merge cursor drains every tick with time <= U (U = the
// current snapshot's UpdateTime) before that snapshot's on_snapshot runs. A
// later-phased instrument's snapshot can therefore deliver a trade stamped
// slightly AFTER this instrument's own snapshot time before on_snapshot sees
// it. The gap must clamp at 0 ("the trade is already known"), never negative.
static void test_trade_gap_skew_clamp() {
  const Symbol inst = make_symbol("510300", 6);
  const Session sess = session_for("sse");
  FactorContext ctx{"20250603", "sse", sess};

  auto reg = make_registry({"trade_gap_ms"}, false);
  CHECK_EQ((int)reg.size(), 1);
  IFactor* gap = reg[0].get();
  CHECK(gap != nullptr);
  gap->on_day_start(ctx);
  gap->on_instrument_day_start(inst);

  BookState book;
  const TsMs t0 = t(9, 30, 0);
  double v;

  // Skewed delivery: a trade stamped t0+500 enters factor state before this
  // instrument's snapshot at t0+200 (another instrument's later UpdateTime
  // raised the merge cutoff past 500 first).
  TickEvent e = trade(t0 + 500, inst, 100, 'B');
  book.apply_trade(e);
  gap->on_tick(e, book);
  Snapshot s1 = make_snap(inst, t0 + 200);
  book.apply_snapshot(s1);
  gap->on_snapshot(s1, book);
  CHECK(gap->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);                    // clamped, not -300

  // The next snapshot restores the ordinary positive gap.
  Snapshot s2 = make_snap(inst, t0 + 3200);
  book.apply_snapshot(s2);
  gap->on_snapshot(s2, book);
  CHECK(gap->value(inst, v));
  CHECK_NEAR(v, 2700.0, 1e-9);                  // 3200 - 500
}

// iter-003 wide-table expansion (#144): snapshot pass-through columns.
static void test_snapshot_passthrough() {
  const Symbol inst = make_symbol("510300", 6);
  const Session sess = session_for("sse");
  FactorContext ctx{"20250603", "sse", sess};

  auto reg = make_registry({"total_bid_vol", "total_ask_vol", "bid_orders5",
                            "ask_orders5", "open_px", "high_px", "low_px",
                            "pre_close_px"}, false);
  CHECK_EQ((int)reg.size(), 8);
  for (auto& f : reg) {
    f->on_day_start(ctx);
    f->on_instrument_day_start(inst);
  }

  BookState book;
  double v;

  // make_snap leaves totals/OHLC unset (0) => NaN; order counts still valid.
  Snapshot s0 = make_snap(inst, t(9, 30, 0));
  book.apply_snapshot(s0);
  for (auto& f : reg) f->on_snapshot(s0, book);
  CHECK(!find(reg, "total_bid_vol")->value(inst, v));
  CHECK(!find(reg, "total_ask_vol")->value(inst, v));
  CHECK(!find(reg, "open_px")->value(inst, v));
  CHECK(!find(reg, "high_px")->value(inst, v));
  CHECK(!find(reg, "low_px")->value(inst, v));
  CHECK(!find(reg, "pre_close_px")->value(inst, v));
  CHECK(find(reg, "bid_orders5")->value(inst, v));
  CHECK_NEAR(v, 7.0, 1e-12);                 // make_snap: 3 + 4x1
  CHECK(find(reg, "ask_orders5")->value(inst, v));
  CHECK_NEAR(v, 6.0, 1e-12);                 // 2 + 4x1

  // Full fields populated: raw totals in fund units, prices in CNY.
  Snapshot s1 = make_snap(inst, t(9, 30, 3));
  s1.total_bid_vol = 12345;
  s1.total_ask_vol = 23456;
  s1.open_px = 3995000;
  s1.high = 4010000;
  s1.low = 3990000;
  s1.pre_close = 3990000;
  book.apply_snapshot(s1);
  for (auto& f : reg) f->on_snapshot(s1, book);
  CHECK(find(reg, "total_bid_vol")->value(inst, v));
  CHECK_NEAR(v, 12345.0, 1e-12);
  CHECK(find(reg, "total_ask_vol")->value(inst, v));
  CHECK_NEAR(v, 23456.0, 1e-12);
  CHECK(find(reg, "open_px")->value(inst, v));
  CHECK_NEAR(v, 3995.0, 1e-12);
  CHECK(find(reg, "high_px")->value(inst, v));
  CHECK_NEAR(v, 4010.0, 1e-12);
  CHECK(find(reg, "low_px")->value(inst, v));
  CHECK_NEAR(v, 3990.0, 1e-12);
  CHECK(find(reg, "pre_close_px")->value(inst, v));
  CHECK_NEAR(v, 3990.0, 1e-12);
}

// iopv_velocity: trailing-60s IOPV change rate (bps/s), span >= 30s guard.
static void test_iopv_velocity() {
  const Symbol inst = make_symbol("510300", 6);
  const Session sess = session_for("sse");
  FactorContext ctx{"20250603", "sse", sess};

  auto reg = make_registry({"iopv_velocity"}, false);
  CHECK_EQ((int)reg.size(), 1);
  IFactor* vel = reg[0].get();
  vel->on_day_start(ctx);
  vel->on_instrument_day_start(inst);

  BookState book;
  const TsMs t0 = t(9, 30, 0);
  double v;

  // IOPV climbs 1 CNY (1000 milli) every 3s from 4000.000.
  for (int k = 0; k <= 20; ++k) {
    Snapshot s = make_snap(inst, t0 + 3000LL * k);
    s.iopv = 4000000 + 1000LL * k;
    s.iopv_valid = true;
    book.apply_snapshot(s);
    vel->on_snapshot(s, book);
    if (k < 10) CHECK(!vel->value(inst, v));  // < 2 points or span < 30s
  }
  // k=20: window holds k=0..20 (span 60s): rel = 20000/4000000 = 50bps / 60s.
  CHECK(vel->value(inst, v));
  CHECK_NEAR(v, 50.0 / 60.0, 1e-9);

  // An invalid-IOPV snapshot emits NaN (and adds no point).
  Snapshot bad = make_snap(inst, t0 + 63000);
  bad.iopv = 0;
  bad.iopv_valid = false;
  book.apply_snapshot(bad);
  vel->on_snapshot(bad, book);
  CHECK(!vel->value(inst, v));
}

// Short-window OFI/imbalance + signed-flow columns: warm-up boundaries,
// exact values, '-' exclusion, empty-window zero-vs-NaN semantics.
static void test_short_window_flow_factors() {
  const Symbol inst = make_symbol("510300", 6);
  const Session sess = session_for("sse");
  FactorContext ctx{"20250603", "sse", sess};

  auto reg = make_registry({"ofi_15s", "ofi_30s", "ofi_60s",
                            "trade_imbalance_15s", "buy_vol_60s", "sell_vol_60s",
                            "large_trade_net_share_60s",
                            "book_event_intensity_60s"}, false);
  CHECK_EQ((int)reg.size(), 8);
  for (auto& f : reg) {
    f->on_day_start(ctx);
    f->on_instrument_day_start(inst);
  }
  IFactor* ofi15 = find(reg, "ofi_15s");
  IFactor* ofi30 = find(reg, "ofi_30s");
  IFactor* ofi60 = find(reg, "ofi_60s");
  IFactor* ti15 = find(reg, "trade_imbalance_15s");
  IFactor* bvol = find(reg, "buy_vol_60s");
  IFactor* svol = find(reg, "sell_vol_60s");
  IFactor* lns = find(reg, "large_trade_net_share_60s");
  IFactor* bei = find(reg, "book_event_intensity_60s");
  CHECK(ofi15 && ofi30 && ofi60 && ti15 && bvol && svol && lns && bei);

  BookState book;
  const TsMs t0 = t(9, 30, 0);
  double v;

  // Snapshot before any event: everything warming up.
  Snapshot s0 = make_snap(inst, t0);
  book.apply_snapshot(s0);
  for (auto& f : reg) f->on_snapshot(s0, book);
  for (auto& f : reg) CHECK(!f->value(inst, v));

  // Deep order (no quote change => OFI contrib 0) + one buy and one sell
  // print. Print prices MUST hit real book levels (buy consumes ask1, sell
  // consumes bid1): an off-book print price would desync the book and OFI
  // would drop those events.
  TickEvent e1 = order(t0 + 1000, inst, Side::Buy, 3988, 100);
  TickEvent e2 = trade(t0 + 2000, inst, 300, 'B');
  e2.price = 4002;                                     // eats ask1: 1000 -> 700
  TickEvent e3 = trade(t0 + 4000, inst, 100, 'S');
  e3.price = 3998;                                     // eats bid1: 2000 -> 1900
  TickEvent evs1[] = {e1, e2, e3};
  for (auto& e : evs1) {
    if (e.is_trade) book.apply_trade(e); else book.apply_order(e);
    for (auto& f : reg) f->on_tick(e, book);
  }

  // t0+15s: 14s since first tick => ofi_15s still warming; same for the rest.
  Snapshot s1 = make_snap(inst, t0 + 15000);
  book.apply_snapshot(s1);
  for (auto& f : reg) f->on_snapshot(s1, book);
  for (auto& f : reg) CHECK(!f->value(inst, v));

  // t0+17s: ofi_15s warm (16s >= 15s); e1 pruned (ts < 2s), e2/e3 kept with
  // contribs +300 (ask1 eaten) and -100 (bid1 eaten) at depths 2700/2600.
  // OFI normalizes by MEAN best depth: 200 / (5300/2).
  // trade_imbalance_15s warm (15s >= 15s): (300-100)/(300+100) = 0.5.
  // Longer windows and the 60s family still warming.
  Snapshot s2 = make_snap(inst, t0 + 17000);
  book.apply_snapshot(s2);
  for (auto& f : reg) f->on_snapshot(s2, book);
  CHECK(ofi15->value(inst, v));
  CHECK_NEAR(v, 400.0 / 5300.0, 1e-12);
  CHECK(!ofi30->value(inst, v));
  CHECK(!ofi60->value(inst, v));
  CHECK(ti15->value(inst, v));
  CHECK_NEAR(v, 0.5, 1e-12);
  CHECK(!bvol->value(inst, v));
  CHECK(!lns->value(inst, v));
  CHECK(!bei->value(inst, v));

  // One more deep order; nothing else yet (timeline stays ordered).
  TickEvent e4 = order(t0 + 61000, inst, Side::Sell, 4012, 100);
  book.apply_order(e4);
  for (auto& f : reg) f->on_tick(e4, book);

  // t0+64.5s: 60s family warm (62.5s+ since first tick/trade/event). The
  // early prints are pruned out of the 60s window (ts < 4.5s), so buy/sell
  // volumes are a legitimate 0.0 while the ratio-style columns go NaN (no
  // attributable flow left); intensity keeps only e4.
  Snapshot s3 = make_snap(inst, t0 + 64500);
  book.apply_snapshot(s3);
  for (auto& f : reg) f->on_snapshot(s3, book);
  CHECK(bvol->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);
  CHECK(svol->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);
  CHECK(!ti15->value(inst, v));                       // window has no signed trade
  CHECK(!lns->value(inst, v));
  CHECK(bei->value(inst, v));
  CHECK_NEAR(v, 1.0 / 60.0, 1e-12);                   // only e4 (61s)
  CHECK(ofi15->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);                          // deep order: zero contrib
  CHECK(ofi30->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);
  CHECK(ofi60->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);

  // Whale buy at ask1 (contrib +200, depth 2800) and one '-' print, which is
  // excluded from the signed family but still an event for intensity.
  TickEvent e5 = trade(t0 + 65000, inst, 200, 'B');
  e5.price = 4002;                                     // eats ask1: 1000 -> 800
  TickEvent e6 = trade(t0 + 65500, inst, 500, '-');
  TickEvent evs2[] = {e5, e6};
  for (auto& e : evs2) {
    book.apply_trade(e);
    for (auto& f : reg) f->on_tick(e, book);
  }

  // t0+66s: whale in the window; '-' print never counts for signed stats.
  Snapshot s4 = make_snap(inst, t0 + 66000);
  book.apply_snapshot(s4);
  for (auto& f : reg) f->on_snapshot(s4, book);
  CHECK(bvol->value(inst, v));
  CHECK_NEAR(v, 200.0, 1e-12);
  CHECK(svol->value(inst, v));
  CHECK_NEAR(v, 0.0, 1e-12);
  CHECK(ti15->value(inst, v));
  CHECK_NEAR(v, 1.0, 1e-12);                          // whale alone in 15s window
  CHECK(lns->value(inst, v));
  CHECK_NEAR(v, 1.0, 1e-12);                          // n=1: top-1 = the whale
  CHECK(bei->value(inst, v));
  CHECK_NEAR(v, 3.0 / 60.0, 1e-12);                   // e4 + e5 + e6
  CHECK(ofi15->value(inst, v));
  CHECK_NEAR(v, 600.0 / 8600.0, 1e-12);               // e4+e5+e6 all inside 15s
  CHECK(ofi60->value(inst, v));
  CHECK_NEAR(v, 600.0 / 8600.0, 1e-12);               // 200 / (8600/3), depths 3000/2800/2800
}

static void test_registry() {
  auto def = make_default_registry();
  CHECK_EQ((int)def.size(), (int)kDefaultFactorNames.size());
  for (std::size_t i = 0; i < def.size(); ++i)
    CHECK(def[i]->name() == kDefaultFactorNames[i]);
  for (auto& f : def) CHECK(!f->is_canary());

  // Unknown name is a hard error (guards the CLI --factors list).
  bool threw = false;
  try {
    make_registry({"no_such_factor"}, false);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);

  // Canaries only appear when explicitly requested.
  auto withc = make_registry({}, true);
  CHECK_EQ((int)withc.size(), (int)kDefaultFactorNames.size() + 2);
  int canaries = 0;
  for (auto& f : withc) canaries += f->is_canary() ? 1 : 0;
  CHECK_EQ(canaries, 2);

  // Hard guard: naming a canary in --factors without the canaries flag is
  // refused (would otherwise smuggle a look-ahead column into output).
  bool canary_refused = false;
  try {
    make_registry({"future_mid_15s"}, false);
  } catch (const std::exception&) {
    canary_refused = true;
  }
  CHECK(canary_refused);
  bool canary_refused2 = false;
  try {
    make_registry({"oir", "future_trade_sign"}, false);
  } catch (const std::exception&) {
    canary_refused2 = true;
  }
  CHECK(canary_refused2);

  // With the flag set, an explicitly named canary is built exactly once
  // (not duplicated by the implicit canary append). A non-empty names list
  // builds ONLY the named factors, so the registry holds the named canary
  // plus the one appended canary it did not name.
  auto withc_named = make_registry({"future_mid_15s"}, true);
  CHECK_EQ((int)withc_named.size(), 2);
  int named_canaries = 0;
  for (auto& f : withc_named)
    named_canaries += (f->name() == "future_mid_15s") ? 1 : 0;
  CHECK_EQ(named_canaries, 1);

  // Wishlist columns are opt-in: buildable via --factors but NOT part of the
  // default registry (already-produced runs keep their sidecars valid).
  static const char* kWishlist[] = {
      "avg_trade_size_60s", "n_trades_60s", "large_trade_share_60s",
      "trade_gap_ms", "cum_trade_vol",
      // iter-003 wide-table expansion (#144)
      "total_bid_vol", "total_ask_vol", "bid_orders5", "ask_orders5",
      "open_px", "high_px", "low_px", "pre_close_px", "iopv_velocity",
      "ofi_15s", "ofi_30s", "trade_imbalance_15s", "trade_imbalance_30s",
      "buy_vol_60s", "sell_vol_60s", "large_trade_net_share_60s",
      "book_event_intensity_60s",
  };
  for (const char* n : kWishlist) {
    auto f = make_snapshot_factor(n);
    if (!f) f = make_tick_factor(n);
    CHECK(f && f->name() == n);
    for (const auto& d : kDefaultFactorNames) CHECK(d != n);
  }
}

int main() {
  test_szse_window_factors();
  test_warmup();
  test_sse_cancel_gating();
  test_trade_window_factors();
  test_trade_gap_skew_clamp();
  test_snapshot_passthrough();
  test_iopv_velocity();
  test_short_window_flow_factors();
  test_registry();
  return hftaft::finish("test_factors");
}
