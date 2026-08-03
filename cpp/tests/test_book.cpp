// test_book.cpp — snapshot anchoring, trade consumption, order insertion,
// cancel handling, divergence detection (unsynced/resync), crossed books.
#include "hftaf/book.hpp"
#include "test_util.hpp"

using namespace hftaf;

namespace {

Snapshot make_snap(TsMs t) {
  Snapshot s;
  s.time = t;
  s.instrument = make_symbol("510300", 6);
  s.last = 4000;
  s.bids[0] = BookLevel{3998, 2000, 3};
  s.bids[1] = BookLevel{3996, 1500, 2};
  s.bids[2] = BookLevel{3994, 1000, 1};
  s.asks[0] = BookLevel{4002, 1000, 2};
  s.asks[1] = BookLevel{4004, 1200, 2};
  s.asks[2] = BookLevel{4006, 1400, 1};
  return s;
}

TickEvent trade(TsMs t, Side side, PriceI price, QtyI vol) {
  TickEvent e;
  e.time = t; e.is_trade = true; e.side = side; e.price = price; e.volume = vol;
  e.trd_bs = side == Side::Buy ? 'B' : side == Side::Sell ? 'S' : '-';
  return e;
}

TickEvent order(TsMs t, Side side, PriceI price, QtyI vol, char type = 'A') {
  TickEvent e;
  e.time = t; e.is_trade = false; e.side = side; e.price = price; e.volume = vol;
  e.ord_type = type;
  return e;
}

}  // namespace

static void test_snapshot_anchor() {
  BookState b;
  CHECK(!b.has_snapshot());
  CHECK(!b.synced());
  CHECK_EQ(b.best_bid_price(), 0);

  b.apply_snapshot(make_snap(34200000));
  CHECK(b.has_snapshot());
  CHECK(b.synced());
  CHECK_EQ(b.best_bid_price(), 3998);
  CHECK_EQ(b.best_bid_qty(), 2000);
  CHECK_EQ(b.best_ask_price(), 4002);
  CHECK_EQ(b.best_ask_qty(), 1000);
  CHECK_EQ(b.last_snapshot_time(), 34200000LL);
  CHECK_EQ((long long)b.resync_count(), 0LL);

  b.reset();
  CHECK(!b.has_snapshot());
  CHECK(!b.synced());
}

static void test_trade_consumption() {
  BookState b;
  b.apply_snapshot(make_snap(34200000));

  // Buyer aggressor partially consumes best ask.
  b.apply_trade(trade(34200500, Side::Buy, 4002, 400));
  CHECK(b.synced());
  CHECK_EQ(b.best_ask_price(), 4002);
  CHECK_EQ(b.best_ask_qty(), 600);

  // Consume the rest of the level: next ask level compacts into slot 0.
  b.apply_trade(trade(34201000, Side::Buy, 4002, 600));
  CHECK(b.synced());
  CHECK_EQ(b.best_ask_price(), 4004);
  CHECK_EQ(b.best_ask_qty(), 1200);

  // Seller aggressor consumes bid depth.
  b.apply_trade(trade(34201500, Side::Sell, 3998, 500));
  CHECK_EQ(b.best_bid_qty(), 1500);

  // '-' print: no aggression, book untouched, no divergence.
  const QtyI q_before = b.best_bid_qty();
  b.apply_trade(trade(34202000, Side::None, 4000, 100));
  CHECK_EQ(b.best_bid_qty(), q_before);
  CHECK(b.synced());

  // Overshoot displayed depth at the trade price => divergence.
  b.apply_trade(trade(34202500, Side::Buy, 4006, 999999));
  CHECK(!b.synced());

  // Events while unsynced are ignored.
  b.apply_trade(trade(34203000, Side::Sell, 3998, 100));
  CHECK_EQ(b.best_bid_qty(), 1500);
  b.apply_order(order(34203500, Side::Buy, 3997, 100));
  CHECK_EQ(b.bids()[1].price, 3996);   // unchanged

  // Next snapshot re-anchors and counts the resync.
  b.apply_snapshot(make_snap(34203000));
  CHECK(b.synced());
  CHECK_EQ((long long)b.resync_count(), 1LL);
  CHECK_EQ(b.best_bid_qty(), 2000);
}

static void test_order_insertion() {
  BookState b;
  b.apply_snapshot(make_snap(34200000));

  // Add at an existing level merges quantity.
  b.apply_order(order(34200500, Side::Buy, 3998, 100));
  CHECK_EQ(b.best_bid_qty(), 2100);
  CHECK_EQ(b.best_bid_price(), 3998);

  // Add between levels keeps price priority.
  b.apply_order(order(34201000, Side::Buy, 3997, 300));
  CHECK_EQ(b.bids()[0].price, 3998);
  CHECK_EQ(b.bids()[1].price, 3997);
  CHECK_EQ(b.bids()[1].volume, 300);
  CHECK_EQ(b.bids()[2].price, 3996);

  // Ask-side insert.
  b.apply_order(order(34201500, Side::Sell, 4003, 250));
  CHECK_EQ(b.asks()[0].price, 4002);
  CHECK_EQ(b.asks()[1].price, 4003);
  CHECK_EQ(b.asks()[2].price, 4004);

  // Events before any snapshot are ignored (no anchor).
  BookState b2;
  b2.apply_order(order(34200500, Side::Buy, 3998, 100));
  b2.apply_trade(trade(34200500, Side::Buy, 4002, 100));
  CHECK(!b2.has_snapshot());
}

static void test_full_book_drop() {
  Snapshot s;
  s.time = 34200000;
  s.instrument = make_symbol("510300", 6);
  s.last = 4000;
  for (int k = 0; k < 10; ++k) {
    s.bids[k] = BookLevel{3998 - 2 * k, 100, 1};
    s.asks[k] = BookLevel{4002 + 2 * k, 100, 1};
  }
  BookState b;
  b.apply_snapshot(s);

  // Worse than all 10 displayed bid levels => invisible, dropped.
  b.apply_order(order(34200500, Side::Buy, 3900, 500));
  CHECK_EQ(b.bids()[9].price, 3998 - 18);
  CHECK_EQ(b.bids()[9].volume, 100);

  // Better than the worst displayed: inserts, level 9 falls off.
  b.apply_order(order(34201000, Side::Buy, 3981, 500));
  CHECK_EQ(b.bids()[9].price, 3981);
  CHECK_EQ(b.bids()[9].volume, 500);
}

static void test_cancels() {
  BookState b;
  b.apply_snapshot(make_snap(34200000));

  // Partial cancel on the ask side ('X' = SZSE cancel marker).
  b.apply_order(order(34200500, Side::Sell, 4004, 200, 'X'));
  CHECK(b.synced());
  CHECK_EQ(b.asks()[1].volume, 1000);

  // Cancel the whole level: it disappears and levels compact.
  b.apply_order(order(34201000, Side::Sell, 4004, 1000, 'X'));
  CHECK_EQ(b.asks()[1].price, 4006);

  // Cancel with unknown size (volume 0) leaves the book untouched.
  const QtyI q = b.best_ask_qty();
  b.apply_order(order(34201500, Side::Sell, 4002, 0, 'X'));
  CHECK_EQ(b.best_ask_qty(), q);

  // Cancelling depth that is no longer displayed is NOT divergence.
  b.apply_order(order(34202000, Side::Sell, 4100, 100, 'X'));
  CHECK(b.synced());

  // 'D' is also treated as cancel.
  b.apply_order(order(34202500, Side::Buy, 3996, 500, 'D'));
  CHECK_EQ(b.bids()[1].volume, 1000);
}

static void test_crossed_book() {
  BookState b;
  b.apply_snapshot(make_snap(34200000));

  // A buy add at/above the best ask crosses the book => unsynced.
  b.apply_order(order(34200500, Side::Buy, 4002, 100));
  CHECK(!b.synced());

  // Snapshot restores sync.
  b.apply_snapshot(make_snap(34203000));
  CHECK(b.synced());
  CHECK_EQ((long long)b.resync_count(), 1LL);
}

int main() {
  test_snapshot_anchor();
  test_trade_consumption();
  test_order_insertion();
  test_full_book_drop();
  test_cancels();
  test_crossed_book();
  return hftaft::finish("test_book");
}
