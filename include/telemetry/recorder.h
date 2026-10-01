#pragma once

// Trace recorder + counters + run metadata.
//
// Trace: each producer thread (workers, control/event thread, bench thread)
// appends to its own preallocated buffer, so recording takes no shared lock.
// The recorder is disabled by default; trace() is then one relaxed load.
//
// Threading contract: enable()/reset()/snapshot()/dump_csv() must only be
// called while no producer is recording (before load_model or after shutdown).

#include "core/status.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace telemetry {

enum class TraceKind : uint8_t {
  Submit = 0,       // a = prompt tokens, b = max_new_tokens
  PrefillStart,     // a = prompt tokens
  PrefillEnd,       // a = 1 if ok
  DecodeStart,
  DecodeEnd,        // b = 1 if ok (token index = order per req_id after PrefillEnd)
  MigrateRequested, // worker = src, a = dst
  MigrateBegin,     // worker = src, a = dst
  MigrateXferPosted, // worker = this worker, a = role (0 send / 1 recv), b = bytes
  MigrateXferDone,  // worker = this worker, a = role, b = 1 if ok
  MigrateCommit,    // worker = new home
  RebalanceDecision, // worker = src, a = dst, b = victim gen_len
  Done,             // a = generated tokens
  Failed,
  Marker,           // free-form bench marker (a, b caller-defined)
  kCount
};

const char* trace_kind_name(TraceKind k);

struct TraceEvent {
  int64_t t_ns = 0; // ns since Recorder epoch (steady_clock)
  TraceKind kind = TraceKind::Marker;
  uint64_t req_id = 0;
  int32_t worker = -1;
  int64_t a = 0;
  int64_t b = 0;
};

struct Counters {
  std::atomic<uint64_t> submits{0};
  std::atomic<uint64_t> migrates_started{0};
  std::atomic<uint64_t> migrates_completed{0};
  std::atomic<uint64_t> migrates_failed{0};
  std::atomic<uint64_t> bytes_migrated{0};
  std::atomic<uint64_t> rebalance_checks{0};
  std::atomic<uint64_t> rebalance_decisions{0};
  std::atomic<uint64_t> slot_rejects{0};
  std::atomic<uint64_t> kv_overflows{0};

  void reset();
  std::vector<std::pair<std::string, uint64_t>> snapshot() const;
};

inline int64_t steady_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class Recorder {
public:
  static Recorder& instance();

  void enable(bool on) { enabled_.store(on, std::memory_order_relaxed); }
  bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

  // Clears all buffers and counters and restarts the epoch.
  void reset();

  // Appends to the calling thread's buffer. No-op when disabled.
  void record(TraceKind kind, uint64_t req_id, int worker, int64_t a, int64_t b);

  // Merged view of every thread's events, sorted by time.
  std::vector<TraceEvent> snapshot() const;
  Status dump_csv(const std::string& path) const;

  int64_t epoch_ns() const { return epoch_ns_.load(std::memory_order_relaxed); }

  Counters counters;

private:
  Recorder();
  struct Impl;
  Impl* impl_;
  std::atomic<bool> enabled_{false};
  std::atomic<int64_t> epoch_ns_{0};
};

inline void trace(TraceKind kind, uint64_t req_id, int worker, int64_t a = 0, int64_t b = 0) {
  Recorder& r = Recorder::instance();
  if (r.enabled())
    r.record(kind, req_id, worker, a, b);
}

inline Counters& counters() { return Recorder::instance().counters; }

// ---- run metadata ----------------------------------------------------------

using MetaKV = std::vector<std::pair<std::string, std::string>>;

// Values of the given env vars (unset ones are reported as "").
MetaKV capture_env(const std::vector<std::string>& names);
// Runs a shell command and returns trimmed stdout ("" on failure). Metadata only.
std::string capture_command(const std::string& cmd);
// Writes a flat JSON object of string values.
Status write_meta_json(const std::string& path, const MetaKV& kv);
// Compile-time git hash (or "unknown").
const char* build_git_hash();

} // namespace telemetry
