// hftaf/engine.cpp — run_job: dual-stream merge, per-instrument state,
// factor sampling, canary delayed attach, label attach, atomic write.
//
// Three execution modes (dispatched at the bottom):
//   raw         — stream tick+snapshot gz, emit factor rows (default).
//   cache-build — stream the same inputs with identical merge/seq semantics
//                 and write a replay cache (verbatim rows of the target
//                 instruments + per-snapshot gap bits). No rows emitted.
//   replay      — recompute factor rows from a cache directory, byte-identical
//                 to the raw run restricted to the cached instruments, without
//                 touching the multi-GB raw gz files.
#include "hftaf/engine.hpp"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
  // Channel membership: set when the instrument shows up in THIS channel's
  // tick stream. The snapshot stream is shared by every channel of the
  // exchange, but each instrument's ticks live in exactly one channel per
  // day; only members may appear in this channel's output (otherwise every
  // ETF would be re-emitted into every channel CSV and the parquet build --
  // which asserts one channel per instrument -- would reject the day).
  bool seen = false;
  // Buffered rows awaiting canary finalization: (row, snapshot). Only used
  // when canaries are present so they can illegitimately read the future.
  std::vector<std::pair<Row, Snapshot>> pending;
};

// Everything the row-emitting pipeline (raw + replay) shares.
struct EmitState {
  const EngineOptions* opts = nullptr;
  const Session* session = nullptr;
  std::vector<std::unique_ptr<IFactor>>* factors = nullptr;
  std::vector<int>* canary_idx = nullptr;
  bool have_canaries = false;
  LabelBuilder* builder = nullptr;
  std::unordered_map<Symbol, InstState, SymbolHash>* insts = nullptr;
  std::vector<Symbol>* inst_order = nullptr;
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

// Largest REQUIRED tick-schema column index; prefix-splitting a row up to here
// decides parse_tick_fields outcome and instrument identity without scanning
// the rest of the row (most channel rows are non-ETF and get skipped).
int tick_required_max(const TickSchema& sc) {
  return std::max({sc.seq, sc.instrument, sc.trade2_order1, sc.transact_time,
                   sc.price, sc.volume, sc.ord_side, sc.ord_type, sc.trd_bs});
}

InstState& get_state(EmitState& es, const Symbol& s) {
  auto it = es.insts->find(s);
  if (it == es.insts->end()) {
    es.insts->emplace(s, InstState{});
    es.inst_order->push_back(s);
    for (auto& f : *es.factors) f->on_instrument_day_start(s);
    return es.insts->find(s)->second;
  }
  return it->second;
}

Row build_row(const EmitState& es, const Snapshot& s) {
  Row r;
  r.date = es.opts->date;
  r.exchange = es.opts->exchange;
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

  r.factors.assign(es.factors->size(), kNan);
  for (std::size_t i = 0; i < es.factors->size(); ++i) {
    if ((*es.factors)[i]->is_canary()) continue;   // filled at delayed finalization
    double v;
    if ((*es.factors)[i]->value(s.instrument, v)) r.factors[i] = v;
  }
  r.fwd_mid.assign(es.opts->horizons_s.size(), kNan);
  r.fwd_last.assign(es.opts->horizons_s.size(), kNan);
  return r;
}

void finalize_pending(EmitState& es, InstState& st, TsMs now) {
  if (!es.have_canaries) return;
  auto& pend = st.pending;
  std::size_t keep = 0;
  for (std::size_t i = 0; i < pend.size(); ++i) {
    if (pend[i].first.time + kCanaryDelayMs <= now) {
      Row r = std::move(pend[i].first);
      Snapshot scopy = pend[i].second;
      for (int ci : *es.canary_idx) {
        double v;
        r.factors[ci] = (*es.factors)[ci]->value_at(r.instrument, r.time, v) ? v : kNan;
      }
      es.builder->push(std::move(r), scopy);
    } else {
      if (keep != i) pend[keep] = std::move(pend[i]);
      ++keep;
    }
  }
  pend.resize(keep);
}

void process_snapshot(EmitState& es, const Snapshot& s, bool& gap_since_snap) {
  // Non-row-producing snapshots (non-ETF, auction) do NOT clear the gap flag:
  // the gap affects each ETF instrument until ITS next snapshot re-anchors it.
  if (!is_etf_code(s.instrument, es.opts->exchange)) return;
  if (!in_continuous_session(*es.session, s.time)) return;
  const bool gap_for_row = gap_since_snap;
  InstState& st = get_state(es, s.instrument);
  // Capture pre-resync divergence before the authoritative re-anchor.
  const bool was_unsynced = st.book.has_snapshot() && !st.book.synced();
  st.book.apply_snapshot(s);
  for (auto& f : *es.factors) f->on_snapshot(s, st.book);

  // Finalize any canary rows old enough now that more future is visible.
  finalize_pending(es, st, s.time);

  Row r = build_row(es, s);
  if (was_unsynced) r.flags |= FLAG_BOOK_UNSYNCED;
  if (gap_for_row) r.flags |= FLAG_SEQ_GAP_BEFORE;

  if (es.have_canaries) {
    st.pending.emplace_back(std::move(r), s);
  } else {
    es.builder->push(std::move(r), s);
  }
  gap_since_snap = false;
}

void process_tick(EmitState& es, const TickEvent& t, std::int64_t& last_seq,
                  bool& gap_since_snap) {
  // SeqNo gap detection runs over the FULL channel stream (all instruments).
  if (last_seq >= 0 && t.seq > last_seq + 1) gap_since_snap = true;
  if (t.seq > last_seq) last_seq = t.seq;

  if (!is_etf_code(t.instrument, es.opts->exchange)) return;
  // Channel membership: this instrument's ticks live in this channel, so it
  // belongs in this channel's output (independent of session filtering).
  get_state(es, t.instrument).seen = true;
  if (!in_continuous_session(*es.session, t.time)) return;
  InstState& st = get_state(es, t.instrument);
  if (t.is_trade) st.book.apply_trade(t);
  else st.book.apply_order(t);
  for (auto& f : *es.factors) f->on_tick(t, st.book);
}

void flush_pending_rows(EmitState& es) {
  // Flush canary pendings (future not available => NaN canary columns).
  for (auto& sym : *es.inst_order) {
    InstState& st = (*es.insts)[sym];
    for (auto& pr : st.pending) {
      Row r = std::move(pr.first);
      Snapshot scopy = pr.second;
      for (int ci : *es.canary_idx) {
        double v;
        r.factors[ci] = (*es.factors)[ci]->value_at(r.instrument, r.time, v) ? v : kNan;
      }
      es.builder->push(std::move(r), scopy);
    }
    st.pending.clear();
  }
  // Flush labels (absent labels stay NaN).
  for (auto& sym : *es.inst_order) es.builder->end_instrument_day(sym);
  es.builder->reset();
}

// Membership filter + deterministic ordering + atomic CSV write + meta.json.
// Shared by the raw and replay modes so both emit identical artifacts.
int finish_output(const JobPaths& paths, const EngineOptions& opts,
                  const std::vector<std::string>& factor_names,
                  std::vector<Row>& collected,
                  const std::unordered_map<Symbol, InstState, SymbolHash>& insts,
                  std::uint64_t tick_bytes, std::uint64_t snap_bytes,
                  std::ostream& log) {
  // --- channel membership filter ---
  // The snapshot stream is shared by every channel of the exchange, but an
  // instrument's ticks live in exactly ONE channel per day. Rows for
  // instruments absent from this channel's tick stream must not leak into
  // this channel's CSV (the parquet build asserts one channel per instrument
  // and would reject the day otherwise).
  collected.erase(std::remove_if(collected.begin(), collected.end(),
                                 [&](const Row& r) {
                                   auto it = insts.find(r.instrument);
                                   return it == insts.end() || !it->second.seen;
                                 }),
                  collected.end());

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
    js << "  \"tick_bytes\": " << tick_bytes << ",\n";
    js << "  \"snapshot_bytes\": " << snap_bytes << ",\n";
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

  log << "ok: rows=" << collected.size() << " instruments=" << insts.size()
      << " factors=" << factor_names.size() << " out=" << paths.out_csv << "\n";
  return 0;
}

// --- registry + builder setup shared by raw and replay ----------------------
struct FactorSetup {
  std::vector<std::unique_ptr<IFactor>> factors;
  std::vector<std::string> factor_names;
  std::vector<int> canary_idx;
  bool have_canaries = false;
};

// Returns false (and logs) on unknown factor names.
bool setup_factors(const EngineOptions& opts, FactorSetup& fs, std::ostream& log) {
  try {
    fs.factors = make_registry(opts.factors, opts.include_canaries);
  } catch (const std::exception& e) {
    log << "error: " << e.what() << "\n";
    return false;
  }
  for (std::size_t i = 0; i < fs.factors.size(); ++i) {
    fs.factor_names.push_back(fs.factors[i]->name());
    if (fs.factors[i]->is_canary()) fs.canary_idx.push_back(static_cast<int>(i));
  }
  fs.have_canaries = !fs.canary_idx.empty();
  return true;
}

// --- minimal JSON field extraction (cache meta.json is written by this engine;
//     no nesting except the instruments array, which these helpers skip) ---
bool json_get_string(const std::string& text, const std::string& key, std::string& out) {
  const std::string pat = "\"" + key + "\"";
  std::size_t p = text.find(pat);
  if (p == std::string::npos) return false;
  p = text.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  p = text.find('"', p + 1);
  if (p == std::string::npos) return false;
  ++p;
  out.clear();
  while (p < text.size()) {
    const char c = text[p];
    if (c == '\\' && p + 1 < text.size()) {
      const char n = text[p + 1];
      switch (n) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        default: out += n; break;
      }
      p += 2;
      continue;
    }
    if (c == '"') return true;
    out += c;
    ++p;
  }
  return false;
}

bool json_get_int(const std::string& text, const std::string& key, std::int64_t& out) {
  const std::string pat = "\"" + key + "\"";
  std::size_t p = text.find(pat);
  if (p == std::string::npos) return false;
  p = text.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < text.size() && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n')) ++p;
  bool neg = false;
  if (p < text.size() && (text[p] == '-' || text[p] == '+')) { neg = (text[p] == '-'); ++p; }
  std::int64_t v = 0;
  bool any = false;
  while (p < text.size() && text[p] >= '0' && text[p] <= '9') {
    v = v * 10 + (text[p] - '0');
    ++p;
    any = true;
  }
  if (!any) return false;
  out = neg ? -v : v;
  return true;
}

bool write_text_atomic(const std::string& path, const std::string& text, std::ostream& log) {
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::trunc | std::ios::binary);
    if (!f.is_open()) { log << "error: cannot write " << tmp << "\n"; return false; }
    f << text;
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) { log << "error: rename failed: " << ec.message() << "\n"; return false; }
  return true;
}

// =============================================================================
// raw mode: stream both gz inputs, emit factor rows.
// =============================================================================
int run_raw(const JobPaths& paths, const EngineOptions& opts, std::ostream& log) {
  const Session session = session_for(opts.exchange);

  // --- registry ---
  FactorSetup fs;
  if (!setup_factors(opts, fs, log)) return 2;

  FactorContext fctx{opts.date, opts.exchange, session};
  for (auto& f : fs.factors) f->on_day_start(fctx);
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
  EmitState es{&opts, &session, &fs.factors, &fs.canary_idx, fs.have_canaries,
               &builder, &insts, &inst_order};

  // Channel-level SeqNo tracking (across ALL instruments, since one channel is a
  // single sequenced stream). A gap flags every row until the next snapshot.
  std::int64_t last_seq = -1;
  bool gap_since_snap = false;

  // --- readers with one-line lookahead ---
  TickEvent cur_tick;
  Snapshot cur_snap;
  bool have_tick = false, have_snap = false;
  std::string line;
  std::vector<std::string_view> pf;

  const std::size_t tick_pref_upto = static_cast<std::size_t>(tick_required_max(tschema));
  auto read_tick = [&]() -> bool {
    while (tick_rd.next_line(line)) {
      if (line.empty()) continue;
      std::string e;
      TickEvent t;
      // Fast path: a prefix split covering the REQUIRED columns decides the
      // accept/reject outcome (optional trailing columns never fail the parse),
      // so the majority of rows -- non-ETF instruments -- skip scanning the
      // rest of the line entirely.
      split_csv_prefix(line, pf, tick_pref_upto);
      if (!parse_tick_fields(tschema, pf, t, e)) continue;   // malformed: skip
      if (is_etf_code(t.instrument, opts.exchange)) {
        // ETF rows also need the optional fields (TrdMoney/OrdNo/BizIndex/...):
        // split the full row and parse again. Cannot fail (all required fields
        // already validated above); the check is belt-and-braces.
        split_csv(line, pf);
        if (!parse_tick_fields(tschema, pf, t, e)) continue;
      }
      cur_tick = t;
      return true;
    }
    return false;
  };

  // parse_snapshot hard-validates only InstrumentID/UpdateTime/LastPrice (every
  // other column goes through lenient opt_* parsers that cannot fail), so a
  // prefix split over those three columns decides accept/reject. The shared
  // snapshot stream is dominated by non-ETF rows (~180 columns); skip them
  // without scanning the rest of the line. A skipped row is kept as a merge
  // barrier at its UpdateTime -- exactly what the old no-op full parse did:
  // process_snapshot returns at the ETF check and does not clear gap state --
  // so merge ordering is unchanged. Rows failing the prefix validity checks
  // fall through to the full parse, which applies the exact legacy semantics.
  const std::size_t snap_pref_upto = static_cast<std::size_t>(
      std::max({sschema.instrument, sschema.update_time, sschema.last}));
  auto read_snap = [&]() -> bool {
    while (snap_rd.next_line(line)) {
      if (line.empty()) continue;
      split_csv_prefix(line, pf, snap_pref_upto);
      if (pf.size() > snap_pref_upto) {
        std::string_view code = trim(pf[sschema.instrument]);
        if (!code.empty() && code.size() <= 12 &&
            !is_etf_code_sv(code, opts.exchange)) {
          TsMs tm;
          PriceI lp;
          if (parse_time_hhmmssmmm(pf[sschema.update_time], tm) &&
              parse_price_milli(pf[sschema.last], lp)) {
            Snapshot s;   // minimal: the merge loop reads .time; the ETF check
                          // in process_snapshot returns immediately.
            s.instrument = make_symbol(code.data(), code.size());
            s.time = tm;
            cur_snap = s;
            return true;
          }
        }
      }
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
      process_tick(es, cur_tick, last_seq, gap_since_snap);
      have_tick = read_tick();
    }
    process_snapshot(es, cur_snap, gap_since_snap);
    have_snap = read_snap();
  }
  // Drain remaining ticks (after the last snapshot) for seq accounting only.
  while (have_tick) {
    if (last_seq >= 0 && cur_tick.seq > last_seq + 1) gap_since_snap = true;
    if (cur_tick.seq > last_seq) last_seq = cur_tick.seq;
    have_tick = read_tick();
  }

  flush_pending_rows(es);
  return finish_output(paths, opts, fs.factor_names, collected, insts,
                       file_size_or(paths.tick_gz, 0), file_size_or(paths.snapshot_gz, 0), log);
}

// =============================================================================
// cache-build mode: one pass over the raw inputs writes a replay cache.
// =============================================================================
// events.csv.gz line grammar (original interleaved merge order):
//   T,<verbatim tick row>                  target instrument tick
//   S,<gap_bit 0|1>,<verbatim snap row>    target snapshot at a row-emission
//                                          site; gap_bit = gap_since_snap at
//                                          that instant (replayed verbatim, so
//                                          replay never re-runs the SeqNo
//                                          machinery over a partial stream)
// Ticks after the last snapshot are NOT cached: the raw drain loop gives them
// seq accounting only (no book/factor/seen effects), so replay must not see them.
int run_cache_build(const JobPaths& paths, const EngineOptions& opts, std::ostream& log) {
  const Session session = session_for(opts.exchange);

  // --- inputs (identical open/schema handling as raw) ---
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

  // --- target set ---
  // Empty list => dynamic: every ETF is cached (membership filtering still
  // happens at replay, exactly as in raw). An explicit list must be given for
  // single-instrument caches: caching "all ETFs" on a channel where an ETF's
  // first snapshot precedes its first tick is only correct if events are kept
  // unconditionally, which the dynamic mode does.
  std::unordered_set<Symbol, SymbolHash> targets;
  for (const auto& c : opts.cache_instruments) {
    if (c.empty() || c.size() > 12) {
      log << "error: bad --cache-instruments entry: '" << c << "'\n";
      return 2;
    }
    targets.insert(make_symbol(c.data(), c.size()));
  }
  const bool dynamic_targets = targets.empty();
  auto is_target = [&](const Symbol& s) -> bool {
    if (!is_etf_code(s, opts.exchange)) return false;
    return dynamic_targets || targets.count(s) > 0;
  };

  // --- event writer ---
  std::error_code ec;
  std::filesystem::create_directories(opts.build_cache_dir, ec);
  const std::string events_path = opts.build_cache_dir + "/events.csv.gz";
  const std::string events_tmp = events_path + ".tmp";
  gzFile ev = gzopen(events_tmp.c_str(), "wb");
  if (!ev) { log << "error: cannot open " << events_tmp << "\n"; return 2; }
  bool write_err = false;
  auto write_event = [&](const char* prefix, std::size_t plen, const std::string& body) {
    if (write_err) return;
    if (gzwrite(ev, prefix, static_cast<unsigned>(plen)) != static_cast<int>(plen)) { write_err = true; return; }
    if (!body.empty() &&
        gzwrite(ev, body.data(), static_cast<unsigned>(body.size())) != static_cast<int>(body.size())) {
      write_err = true;
      return;
    }
    if (gzwrite(ev, "\n", 1) != 1) write_err = true;
  };

  std::int64_t n_tick_events = 0, n_snap_events = 0;
  std::set<Symbol> member_syms;   // targets with >=1 cached event (sorted for meta)

  // --- merge state: identical accounting to raw ---
  std::int64_t last_seq = -1;
  bool gap_since_snap = false;

  TickEvent cur_tick;
  Snapshot cur_snap;
  std::string cur_tick_line, cur_snap_line;
  bool have_tick = false, have_snap = false;
  std::string line;
  std::vector<std::string_view> pf;

  const std::size_t tick_pref_upto = static_cast<std::size_t>(tick_required_max(tschema));
  auto read_tick = [&]() -> bool {
    while (tick_rd.next_line(line)) {
      if (line.empty()) continue;
      std::string e;
      TickEvent t;
      // Build mode never needs the optional tick fields (no factors run here);
      // the prefix parse fully decides validity + seq/time/instrument.
      split_csv_prefix(line, pf, tick_pref_upto);
      if (!parse_tick_fields(tschema, pf, t, e)) continue;
      cur_tick = t;
      cur_tick_line = line;
      return true;
    }
    return false;
  };

  const std::size_t snap_pref_upto = static_cast<std::size_t>(
      std::max({sschema.instrument, sschema.update_time, sschema.last}));
  auto read_snap = [&]() -> bool {
    while (snap_rd.next_line(line)) {
      if (line.empty()) continue;
      // Same barrier fast path as raw mode (see run_raw for the argument).
      split_csv_prefix(line, pf, snap_pref_upto);
      if (pf.size() > snap_pref_upto) {
        std::string_view code = trim(pf[sschema.instrument]);
        if (!code.empty() && code.size() <= 12 &&
            !is_etf_code_sv(code, opts.exchange)) {
          TsMs tm;
          PriceI lp;
          if (parse_time_hhmmssmmm(pf[sschema.update_time], tm) &&
              parse_price_milli(pf[sschema.last], lp)) {
            Snapshot s;
            s.instrument = make_symbol(code.data(), code.size());
            s.time = tm;
            cur_snap = s;
            cur_snap_line.clear();   // barrier: never written to the cache
            return true;
          }
        }
      }
      std::string e;
      Snapshot s;
      if (parse_snapshot(sschema, line, s, e)) { cur_snap = s; cur_snap_line = line; return true; }
    }
    return false;
  };

  have_tick = read_tick();
  have_snap = read_snap();
  while (have_snap) {
    const TsMs U = cur_snap.time;
    while (have_tick && cur_tick.time <= U) {
      if (last_seq >= 0 && cur_tick.seq > last_seq + 1) gap_since_snap = true;
      if (cur_tick.seq > last_seq) last_seq = cur_tick.seq;
      if (is_target(cur_tick.instrument)) {
        write_event("T,", 2, cur_tick_line);
        ++n_tick_events;
        member_syms.insert(cur_tick.instrument);
      }
      have_tick = read_tick();
    }
    // Mirror process_snapshot's gating: a snapshot clears the gap flag only at
    // a row-emission site (ETF + continuous session).
    const bool row_site = is_etf_code(cur_snap.instrument, opts.exchange) &&
                          in_continuous_session(session, cur_snap.time);
    if (row_site) {
      if (is_target(cur_snap.instrument)) {
        if (gap_since_snap) write_event("S,1,", 4, cur_snap_line);
        else write_event("S,0,", 4, cur_snap_line);
        ++n_snap_events;
        member_syms.insert(cur_snap.instrument);
      }
      gap_since_snap = false;
    }
    have_snap = read_snap();
  }
  // Tail ticks get seq accounting only in raw mode => not cached.
  while (have_tick) {
    if (last_seq >= 0 && cur_tick.seq > last_seq + 1) gap_since_snap = true;
    if (cur_tick.seq > last_seq) last_seq = cur_tick.seq;
    have_tick = read_tick();
  }

  if (gzclose(ev) != Z_OK) write_err = true;
  if (write_err) {
    log << "error: failed writing cache events to " << events_tmp << "\n";
    std::filesystem::remove(events_tmp, ec);
    return 2;
  }
  std::filesystem::rename(events_tmp, events_path, ec);
  if (ec) { log << "error: rename failed: " << ec.message() << "\n"; return 2; }

  // --- meta.json ---
  {
    std::ostringstream js;
    js << "{\n";
    js << "  \"kind\": \"hftaf-cache\",\n";
    js << "  \"version\": 1,\n";
    js << "  \"build_id\": \"";
    { std::string esc; json_escape(esc, opts.build_id); js << esc; }
    js << "\",\n";
    js << "  \"exchange\": \"" << opts.exchange << "\",\n";
    js << "  \"date\": \"" << opts.date << "\",\n";
    js << "  \"channel\": " << opts.channel << ",\n";
    js << "  \"tick_header\": \"";
    { std::string esc; json_escape(esc, tick_header); js << esc; }
    js << "\",\n";
    js << "  \"snapshot_header\": \"";
    { std::string esc; json_escape(esc, snap_header); js << esc; }
    js << "\",\n";
    js << "  \"tick_bytes\": " << file_size_or(paths.tick_gz, 0) << ",\n";
    js << "  \"snapshot_bytes\": " << file_size_or(paths.snapshot_gz, 0) << ",\n";
    js << "  \"tick_events\": " << n_tick_events << ",\n";
    js << "  \"snap_events\": " << n_snap_events << ",\n";
    js << "  \"instruments\": [";
    bool first = true;
    for (const auto& s : member_syms) {
      if (!first) js << ", ";
      js << "\"" << symbol_to_string(s) << "\"";
      first = false;
    }
    js << "]\n";
    js << "}\n";
    if (!write_text_atomic(opts.build_cache_dir + "/meta.json", js.str(), log)) return 2;
  }

  log << "ok: cache built dir=" << opts.build_cache_dir
      << " tick_events=" << n_tick_events << " snap_events=" << n_snap_events
      << " instruments=" << member_syms.size() << "\n";
  return 0;
}

// =============================================================================
// replay mode: recompute factor rows from a cache directory.
// =============================================================================
// Assumes factors keep PER-INSTRUMENT state only (true for the whole v1
// registry; cross-asset factors are deferred by design). A factor that ever
// reads other instruments' events would need every such instrument cached.
int run_replay(const JobPaths& paths, const EngineOptions& opts, std::ostream& log) {
  // --- load + cross-check cache meta ---
  const std::string meta_path = opts.use_cache_dir + "/meta.json";
  std::string meta_text;
  {
    std::ifstream mf(meta_path);
    if (!mf.is_open()) { log << "error: cannot open cache meta: " << meta_path << "\n"; return 2; }
    std::ostringstream mss;
    mss << mf.rdbuf();
    meta_text = mss.str();
  }
  std::string kind, c_exchange, c_date, tick_header, snap_header;
  std::int64_t c_channel = -1, c_tick_bytes = 0, c_snap_bytes = 0;
  if (!json_get_string(meta_text, "kind", kind) || kind != "hftaf-cache") {
    log << "error: " << meta_path << " is not an hftaf cache\n";
    return 2;
  }
  if (!json_get_string(meta_text, "exchange", c_exchange) || c_exchange != opts.exchange) {
    log << "error: cache exchange mismatch (cache='" << c_exchange
        << "', requested='" << opts.exchange << "')\n";
    return 2;
  }
  if (!json_get_string(meta_text, "date", c_date) || c_date != opts.date) {
    log << "error: cache date mismatch (cache='" << c_date
        << "', requested='" << opts.date << "')\n";
    return 2;
  }
  if (!json_get_int(meta_text, "channel", c_channel) || c_channel != opts.channel) {
    log << "error: cache channel mismatch (cache=" << c_channel
        << ", requested=" << opts.channel << ")\n";
    return 2;
  }
  if (!json_get_string(meta_text, "tick_header", tick_header) ||
      !json_get_string(meta_text, "snapshot_header", snap_header)) {
    log << "error: cache meta lacks header fields\n";
    return 2;
  }
  json_get_int(meta_text, "tick_bytes", c_tick_bytes);
  json_get_int(meta_text, "snapshot_bytes", c_snap_bytes);

  const Session session = session_for(opts.exchange);

  // --- registry (same as raw) ---
  FactorSetup fs;
  if (!setup_factors(opts, fs, log)) return 2;

  FactorContext fctx{opts.date, opts.exchange, session};
  for (auto& f : fs.factors) f->on_day_start(fctx);
  if (!cancel_decode_reliable(opts.exchange)) {
    log << "warning: cancel decode unreliable for exchange '" << opts.exchange
        << "'; order_arrival_60s/cancel_ratio_60s will emit NaN\n";
  }

  // --- schemas from the cached original headers ---
  std::vector<std::string_view> hf, hs;
  split_csv(tick_header, hf);
  split_csv(snap_header, hs);
  TickSchema tschema;
  SnapshotSchema sschema;
  std::string err;
  if (!make_tick_schema(hf, tschema, err)) { log << "error: " << err << "\n"; return 2; }
  if (!make_snapshot_schema(hs, sschema, err)) { log << "error: " << err << "\n"; return 2; }

  // --- label builder + per-instrument state (same as raw) ---
  std::vector<Row> collected;
  LabelBuilder builder(LabelConfig{opts.horizons_s}, session);
  builder.set_sink([&](Row&& r) { collected.push_back(std::move(r)); });

  std::unordered_map<Symbol, InstState, SymbolHash> insts;
  std::vector<Symbol> inst_order;
  EmitState es{&opts, &session, &fs.factors, &fs.canary_idx, fs.have_canaries,
               &builder, &insts, &inst_order};

  // Cached ticks are target-instrument only, so this seq accounting runs over
  // a partial stream and is meaningless -- but harmless: every S event
  // overwrites gap_since_snap with the recorded bit before process_snapshot
  // reads it, and nothing else observes these values.
  std::int64_t last_seq = -1;
  bool gap_since_snap = false;

  // --- event stream ---
  const std::string events_path = opts.use_cache_dir + "/events.csv.gz";
  GzLineReader ev(events_path);
  if (!ev.ok()) { log << "error: " << ev.error() << "\n"; return 2; }
  std::string line;
  std::string perr;
  while (ev.next_line(line)) {
    if (line.size() < 2 || line[1] != ',') continue;   // malformed: skip
    if (line[0] == 'T') {
      TickEvent t;
      if (!parse_tick(tschema, std::string_view(line).substr(2), t, perr)) continue;
      process_tick(es, t, last_seq, gap_since_snap);
    } else if (line[0] == 'S' && line.size() >= 4 && line[3] == ',') {
      const bool gap_bit = (line[2] == '1');
      Snapshot s;
      if (!parse_snapshot(sschema, std::string_view(line).substr(4), s, perr)) continue;
      gap_since_snap = gap_bit;
      process_snapshot(es, s, gap_since_snap);
    }
    // Unknown tag: malformed cache line; skip.
  }

  flush_pending_rows(es);
  return finish_output(paths, opts, fs.factor_names, collected, insts,
                       static_cast<std::uint64_t>(c_tick_bytes),
                       static_cast<std::uint64_t>(c_snap_bytes), log);
}

}  // namespace

int run_job(const JobPaths& paths, const EngineOptions& opts, std::ostream& log) {
  if (opts.exchange != "sse" && opts.exchange != "szse") {
    log << "error: exchange must be sse|szse, got '" << opts.exchange << "'\n";
    return 2;
  }
  if (!opts.use_cache_dir.empty() && !opts.build_cache_dir.empty()) {
    log << "error: build_cache_dir and use_cache_dir are mutually exclusive\n";
    return 2;
  }
  if (!opts.use_cache_dir.empty()) return run_replay(paths, opts, log);
  if (!opts.build_cache_dir.empty()) return run_cache_build(paths, opts, log);
  return run_raw(paths, opts, log);
}

}  // namespace hftaf
