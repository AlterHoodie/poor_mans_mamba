#include "telemetry/recorder.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

#ifndef MAMBASERVE_GIT_HASH
#define MAMBASERVE_GIT_HASH "unknown"
#endif

namespace telemetry {

namespace {
constexpr size_t kInitialReserve = 1 << 16;

thread_local std::vector<TraceEvent>* tl_buf = nullptr;
} // namespace

const char* trace_kind_name(TraceKind k) {
  switch (k) {
  case TraceKind::Submit:
    return "Submit";
  case TraceKind::PrefillStart:
    return "PrefillStart";
  case TraceKind::PrefillEnd:
    return "PrefillEnd";
  case TraceKind::DecodeStart:
    return "DecodeStart";
  case TraceKind::DecodeEnd:
    return "DecodeEnd";
  case TraceKind::MigrateRequested:
    return "MigrateRequested";
  case TraceKind::MigrateBegin:
    return "MigrateBegin";
  case TraceKind::MigrateXferPosted:
    return "MigrateXferPosted";
  case TraceKind::MigrateXferDone:
    return "MigrateXferDone";
  case TraceKind::MigrateCommit:
    return "MigrateCommit";
  case TraceKind::RebalanceDecision:
    return "RebalanceDecision";
  case TraceKind::Done:
    return "Done";
  case TraceKind::Failed:
    return "Failed";
  case TraceKind::Marker:
    return "Marker";
  case TraceKind::kCount:
    break;
  }
  return "Unknown";
}

void Counters::reset() {
  submits = 0;
  migrates_started = 0;
  migrates_completed = 0;
  migrates_failed = 0;
  bytes_migrated = 0;
  rebalance_checks = 0;
  rebalance_decisions = 0;
  slot_rejects = 0;
  kv_overflows = 0;
}

std::vector<std::pair<std::string, uint64_t>> Counters::snapshot() const {
  return {
      {"submits", submits.load()},
      {"migrates_started", migrates_started.load()},
      {"migrates_completed", migrates_completed.load()},
      {"migrates_failed", migrates_failed.load()},
      {"bytes_migrated", bytes_migrated.load()},
      {"rebalance_checks", rebalance_checks.load()},
      {"rebalance_decisions", rebalance_decisions.load()},
      {"slot_rejects", slot_rejects.load()},
      {"kv_overflows", kv_overflows.load()},
  };
}

struct Recorder::Impl {
  mutable std::mutex mu; // guards `buffers` (registration + reset/snapshot only)
  std::vector<std::unique_ptr<std::vector<TraceEvent>>> buffers;
};

Recorder::Recorder() : impl_(new Impl) { epoch_ns_.store(steady_now_ns()); }

Recorder& Recorder::instance() {
  static Recorder* r = new Recorder(); // intentionally leaked: outlives worker threads
  return *r;
}

void Recorder::reset() {
  std::lock_guard<std::mutex> lk(impl_->mu);
  for (auto& b : impl_->buffers)
    b->clear(); // capacity (and the thread_local pointers) stay valid
  counters.reset();
  epoch_ns_.store(steady_now_ns());
}

void Recorder::record(TraceKind kind, uint64_t req_id, int worker, int64_t a, int64_t b) {
  if (!tl_buf) {
    auto buf = std::make_unique<std::vector<TraceEvent>>();
    buf->reserve(kInitialReserve);
    tl_buf = buf.get();
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->buffers.push_back(std::move(buf));
  }
  tl_buf->push_back(TraceEvent{
      .t_ns = steady_now_ns() - epoch_ns_.load(std::memory_order_relaxed),
      .kind = kind,
      .req_id = req_id,
      .worker = static_cast<int32_t>(worker),
      .a = a,
      .b = b,
  });
}

std::vector<TraceEvent> Recorder::snapshot() const {
  std::vector<TraceEvent> out;
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    size_t n = 0;
    for (const auto& b : impl_->buffers)
      n += b->size();
    out.reserve(n);
    for (const auto& b : impl_->buffers)
      out.insert(out.end(), b->begin(), b->end());
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const TraceEvent& x, const TraceEvent& y) { return x.t_ns < y.t_ns; });
  return out;
}

Status Recorder::dump_csv(const std::string& path) const {
  std::ofstream f(path);
  if (!f)
    return Status::RuntimeError("cannot open trace csv for writing: " + path);
  f << "t_ns,kind,req_id,worker,a,b\n";
  for (const TraceEvent& e : snapshot()) {
    f << e.t_ns << ',' << trace_kind_name(e.kind) << ',' << e.req_id << ',' << e.worker << ','
      << e.a << ',' << e.b << '\n';
  }
  return f.good() ? Status::Ok() : Status::RuntimeError("failed writing trace csv: " + path);
}

// ---- run metadata ----------------------------------------------------------

MetaKV capture_env(const std::vector<std::string>& names) {
  MetaKV out;
  for (const std::string& n : names) {
    const char* v = std::getenv(n.c_str());
    out.emplace_back(n, v ? v : "");
  }
  return out;
}

std::string capture_command(const std::string& cmd) {
  std::string out;
  FILE* p = popen((cmd + " 2>/dev/null").c_str(), "r");
  if (!p)
    return out;
  char buf[256];
  while (fgets(buf, sizeof(buf), p))
    out += buf;
  pclose(p);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
    out.pop_back();
  return out;
}

namespace {
std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
    case '"':
      o += "\\\"";
      break;
    case '\\':
      o += "\\\\";
      break;
    case '\n':
      o += "\\n";
      break;
    case '\r':
      o += "\\r";
      break;
    case '\t':
      o += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", c);
        o += buf;
      } else {
        o += c;
      }
    }
  }
  return o;
}
} // namespace

Status write_meta_json(const std::string& path, const MetaKV& kv) {
  std::ofstream f(path);
  if (!f)
    return Status::RuntimeError("cannot open meta json for writing: " + path);
  f << "{\n";
  for (size_t i = 0; i < kv.size(); ++i) {
    f << "  \"" << json_escape(kv[i].first) << "\": \"" << json_escape(kv[i].second) << "\""
      << (i + 1 < kv.size() ? "," : "") << "\n";
  }
  f << "}\n";
  return f.good() ? Status::Ok() : Status::RuntimeError("failed writing meta json: " + path);
}

const char* build_git_hash() { return MAMBASERVE_GIT_HASH; }

} // namespace telemetry
