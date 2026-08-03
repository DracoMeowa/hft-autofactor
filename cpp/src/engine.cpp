// hftaf/engine.cpp — run_job: dual-stream merge, per-instrument state,
// factor sampling, canary delayed attach, label attach, atomic write.
#include "hftaf/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hftaf/book.hpp"
#include "hftaf/decode.hpp"
#include "hftaf/factors.hpp"
#include "hftaf/io.hpp"
#include "hftaf/labels.hpp"
#include "hftaf/output.hpp"
#include "hftaf/session.hpp"
#include "hftaf/types.hpp"

namespace hftaf {
namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr TsMs kCanaryDelayMs = 15000;   // canaries peek 15s ahead

double milli_to_cny(PriceI p) { return static_cast<double>(p) / 1000.0; }

struct InstState {
  BookState book;
  bool seen = false;
  // Buffered rows awaiting canary finalization: (row, snapshot). Only used
  // when canaries are present so they can illegitimately read the future.
  std::vector<std::pair<Row, Snapshot>> pending;
};

std::uint64_t file_size_or(const std::string& p, std::uint64_t fallback) {
  std::error_code ec;
  const auto sz = std::filesystem::file_size(p, ec);
  if (ec) return fallback;
  return static_cast<std::uint64_t>(sz);
}

void json_escape(std::string& out, const std::string& s) {
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default: out += c;
    }
  }
}

}  // namespace

int run_job(const JobPaths& paths, const EngineOptions& opts, std::ostream& log) {
  if (opts.exchange != "sse" && opts.exchange != "szse") {
    log << "error: exchange must be sse|szse, got '" << opts.exchange << "'\n";
    return 2;
  }
  const Session session = session_for(opts.exchange);

  // --- registry ---
  std::vector<std::unique_ptr<IFactor>> factors;
  try {
    factors = make_registry(opts.factors, opts.include_canaries);
  } catch (const std::exception& e) {
    log << "error: " << e.what() << "\n";
    return 2;
  }
  std::vector<std::string> factor_names;
  std::vector<int> canary_idx;
  for (std::size_t i = 0; i < factors.size(); ++i) {
    factor_names.push_back(factors[i]->name());
    if (factors[i]->is_canary()) canary_idx.push_back(static_cast<int>(i));
  }
  const bool have_canaries = !canary_idx.empty();

  FactorContext fctx{opts.date, opts.exchange, session};
  for (auto& f : factors) f->on_day_start(fctx);
  if (!cancel_decode_reliable(opts.exchange)) {
    log << "warning: cancel decode unreliable for exchange '" << opts.exchange
        << "'; order_arrival_60s/cancel_ratio_60s will emit NaN\n";
  }

  // --- inputs ---
  GzLineReader tick_rd(paths.tick_gz);
  GzLineReader snap_rd(paths.snapshot_gz);
  if (!tick_rd.ok()) { log << "error: " << tick_rd.error() << "\n"; return 2; }
  if (!snap_rd.ok()) { log << "error: " << snap_rd.error() << "\n"; return 2; }

  std::string tick_header, snap_header;
  if (!tick_rd.next_line(tick_header)) { log << "error: empty tick file\n"; return 2; }
  if (!snap_rd.next_line(snap_header)) { log << "error: empty snapshot file\n"; return 2; }

  std::vector<std::string_view> hf, hs;
  split_csv(tick_header, hf);
  split_csv(snap_header, hs);
  TickSchema tschema;
  SnapshotSchema sschema;
  std::string err;
  if (!make_tick_schema(hf, tschema, err)) { log << "error: " << err << "\n"; return 2; }
  if (!make_snapshot_schema(hs, sschema, err)) { log << "error: " << err << "\n"; return 2; }

  // --- label builder collects completed rows ---
  std::vector<Row> collected;
  LabelBuilder builder(LabelConfig{opts.horizons_s}, session);
  builder.set_sink([&](Row&& r) { collected.push_back(std::move(r)); });

  // --- per-instrument state ---
  std::unordered_map<Symbol, InstState, SymbolHash> insts;
  std::vector<Symbol> inst_order;

  auto get_state = [&](const Symbol& s) -> InstState& {
    auto it = insts.find(s);
    if (it == insts.end()) {
      insts.emplace(s, InstState{});
      inst_order.push_back(s);
      for (auto& f : factors) f->on_instrument_day_start(s);
      return insts.find(s)->second;
    }
    return it->second;
  };

  // Channel-level SeqNo tracking (across ALL instruments, since one channel is a
  // single sequenced stream). A gap flags every row until the next snapshot.
  std::int64_t last_seq = -1;
  bool gap_since_snap = false;

  auto build_row = [&](const Snapshot& s) -> Row {
    Row r;
    r.date = opts.date;
    r.exchange = opts.exchange;
    r.instrument = s.instrument;
    r.time = s.time;
    r.snap_seq = s.seq;
    r.flags = 0;
    if (!s.iopv_valid) r.flags |= FLAG_IOPV_INVALID;
    const BookLevel& b1 = s.bids[0];
    const BookLevel& a1 = s.asks[0];
    if (b1.price <= 0 || a1.price <= 0) r.flags |= FLAG_ONE_SIDED_BOOK;

    r.bid1_px = b1.price > 0 ? milli_to_cny(b1.price) : kNan;
    r.ask1_px = a1.price > 0 ? milli_to_cny(a1.price) : kNan;
    r.bid1_qty = b1.volume;
    r.ask1_qty = a1.volume;
    r.mid_px = (b1.price > 0 && a1.price > 0)
                   ? (static_cast<double>(b1.price) + static_cast<double>(a1.price)) / 2000.0
                   : kNan;
    r.last_px = s.last > 0 ? milli_to_cny(s.last) : kNan;
    QtyI db5 = 0, da5 = 0;
    for (int k = 0; k < 5; ++k) {
      if (s.bids[k].price > 0) db5 += s.bids[k].volume;
      if (s.asks[k].price > 0) da5 += s.asks[k].volume;
    }
    r.depth_bid5 = db5;
    r.depth_ask5 = da5;

    r.factors.assign(factors.size(), kNan);
    for (std::size_t i = 0; i < factors.size(); ++i) {
      if (factors[i]->is_canary()) continue;   // filled at delayed finalization
      double v;
      if (factors[i]->value(s.instrument, v)) r.factors[i] = v;
    }
    r.fwd_mid.assign(opts.horizons_s.size(), kNan);
    r.fwd_last.assign(opts.horizons_s.size(), kNan);
    return r;
  };

  auto finalize_pending = [&](InstState& st, TsMs now) {
    if (!have_canaries) return;
    auto& pend = st.pending;
    std::size_t keep = 0;
    for (std::size_t i = 0; i < pend.size(); ++i) {
      if (pend[i].first.time + kCanaryDelayMs <= now) {
        Row r = std::move(pend[i].first);
        Snapshot scopy = pend[i].second;
        for (int ci : canary_idx) {
          double v;
          r.factors[ci] = factors[ci]->value_at(r.instrument, r.time, v) ? v : kNan;
        }
        builder.push(std::move(r), scopy);
      } else {
        if (keep != i) pend[keep] = std::move(pend[i]);
        ++keep;
      }
    }
    pend.resize(keep);
  };

  auto process_snapshot = [&](const Snapshot& s) {
    // Non-row-producing snapshots (non-ETF, auction) do NOT clear the gap flag:
    // the gap affects each ETF instrument until ITS next snapshot re-anchors it.
    if (!is_etf_code(s.instrument, opts.exchange)) return;
    if (!in_continuous_session(session, s.time)) return;
    const bool gap_for_row = gap_since_snap;
    InstState& st = get_state(s.instrument);
    // Capture pre-resync divergence before the authoritative re-anchor.
    const bool was_unsynced = st.book.has_snapshot() && !st.book.synced();
    st.book.apply_snapshot(s);
    for (auto& f : factors) f->on_snapshot(s, st.book);

    // Finalize any canary rows old enough now that more future is visible.
    finalize_pending(st, s.time);

    Row r = build_row(s);
    if (was_unsynced) r.flags |= FLAG_BOOK_UNSYNCED;
    if (gap_for_row) r.flags |= FLAG_SEQ_GAP_BEFORE;

    if (have_canaries) {
      st.pending.emplace_back(std::move(r), s);
    } else {
      builder.push(std::move(r), s);
    }
    gap_since_snap = false;
  };

  auto process_tick = [&](const TickEvent& t) {
    // SeqNo gap detection runs over the FULL channel stream (all instruments).
    if (last_seq >= 0 && t.seq > last_seq + 1) gap_since_snap = true;
    if (t.seq > last_seq) last_seq = t.seq;

    if (!is_etf_code(t.instrument, opts.exchange)) return;
    if (!in_continuous_session(session, t.time)) return;
    InstState& st = get_state(t.instrument);
    if (t.is_trade) st.book.apply_trade(t);
    else st.book.apply_order(t);
    for (auto& f : factors) f->on_tick(t, st.book);
  };

  // --- readers with one-line lookahead ---
  TickEvent cur_tick;
  Snapshot cur_snap;
  bool have_tick = false, have_snap = false;
  std::string line;

  auto read_tick = [&]() -> bool {
    while (tick_rd.next_line(line)) {
      if (line.empty()) continue;
      std::string e;
      TickEvent t;
      if (parse_tick(tschema, line, t, e)) { cur_tick = t; return true; }
      // Malformed line: skip but keep the stream moving.
    }
    return false;
  };
  auto read_snap = [&]() -> bool {
    while (snap_rd.next_line(line)) {
      if (line.empty()) continue;
      std::string e;
      Snapshot s;
      if (parse_snapshot(sschema, line, s, e)) { cur_snap = s; return true; }
    }
    return false;
  };

  have_tick = read_tick();
  have_snap = read_snap();

  // Merge: all ticks with TransactTime <= U are processed before snapshot U.
  while (have_snap) {
    const TsMs U = cur_snap.time;
    while (have_tick && cur_tick.time <= U) {
      process_tick(cur_tick);
      have_tick = read_tick();
    }
    process_snapshot(cur_snap);
    have_snap = read_snap();
  }
  // Drain remaining ticks (after the last snapshot) for seq accounting only.
  while (have_tick) {
    if (last_seq >= 0 && cur_tick.seq > last_seq + 1) gap_since_snap = true;
    if (cur_tick.seq > last_seq) last_seq = cur_tick.seq;
    have_tick = read_tick();
  }

  // --- flush canary pendings (future not available => NaN canary columns) ---
  for (auto& sym : inst_order) {
    InstState& st = insts[sym];
    for (auto& pr : st.pending) {
      Row r = std::move(pr.first);
      Snapshot scopy = pr.second;
      for (int ci : canary_idx) {
        double v;
        r.factors[ci] = factors[ci]->value_at(r.instrument, r.time, v) ? v : kNan;
      }
      builder.push(std::move(r), scopy);
    }
    st.pending.clear();
  }
  // --- flush labels (absent labels stay NaN) ---
  for (auto& sym : inst_order) builder.end_instrument_day(sym);
  builder.reset();

  // --- deterministic ordering: instrument asc, then time asc ---
  std::sort(collected.begin(), collected.end(), [](const Row& a, const Row& b) {
    if (a.instrument < b.instrument) return true;
    if (b.instrument < a.instrument) return false;
    return a.time < b.time;
  });

  // --- write <out>.tmp then atomic rename ---
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(paths.out_csv).parent_path(), ec);
  const std::string tmp_path = paths.out_csv + ".tmp";
  {
    OutputWriter w(tmp_path, factor_names, opts.horizons_s);
    if (!w.ok()) { log << "error: " << w.error() << "\n"; return 2; }
    w.write_header();
    for (const auto& r : collected) w.write_row(r);
    w.finish();
  }
  std::filesystem::rename(tmp_path, paths.out_csv, ec);
  if (ec) { log << "error: rename failed: " << ec.message() << "\n"; return 2; }

  // --- meta.json sidecar ---
  {
    std::ostringstream js;
    js << "{\n";
    js << "  \"build_id\": \"";
    { std::string esc; json_escape(esc, opts.build_id); js << esc; }
    js << "\",\n";
    js << "  \"exchange\": \"" << opts.exchange << "\",\n";
    js << "  \"date\": \"" << opts.date << "\",\n";
    js << "  \"channel\": " << opts.channel << ",\n";
    js << "  \"rows\": " << collected.size() << ",\n";
    js << "  \"tick_bytes\": " << file_size_or(paths.tick_gz, 0) << ",\n";
    js << "  \"snapshot_bytes\": " << file_size_or(paths.snapshot_gz, 0) << ",\n";
    js << "  \"factors\": [";
    for (std::size_t i = 0; i < factor_names.size(); ++i) {
      if (i) js << ", ";
      js << "\"" << factor_names[i] << "\"";
    }
    js << "],\n";
    js << "  \"horizons_s\": [";
    for (std::size_t i = 0; i < opts.horizons_s.size(); ++i) {
      if (i) js << ", ";
      js << opts.horizons_s[i];
    }
    js << "]\n";
    js << "}\n";
    std::ofstream mf(paths.out_csv + ".meta.json", std::ios::trunc);
    mf << js.str();
  }

  log << "ok: rows=" << collected.size() << " instruments=" << inst_order.size()
      << " factors=" << factor_names.size() << " out=" << paths.out_csv << "\n";
  return 0;
}

}  // namespace hftaf
