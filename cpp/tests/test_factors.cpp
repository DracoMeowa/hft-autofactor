// test_factors.cpp — in-memory registry checks: exact values for snapshot
// factors, 60s window warm-up, SZSE cancel-gated factors, OFI zero-flow case,
// canary look-ahead semantics, causal value_at() refusal, registry errors.
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
}

int main() {
  test_szse_window_factors();
  test_warmup();
  test_sse_cancel_gating();
  test_registry();
  return hftaft::finish("test_factors");
}
