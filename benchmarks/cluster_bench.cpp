// cluster_bench: drives ClusterScheduler end to end and derives latency /
// throughput / migration metrics from the telemetry trace.
//
// Scenarios (--scenario scale|migrate|rebalance|all, comma separated):
//   scale      A. replica scaling: workers x concurrent sessions, no migration
//   migrate    B. forced migrations of one session at chosen decode steps,
//              with the rest of the cluster idle or busy (interference)
//   rebalance  C. skewed load with the rebalance policy off/on
//
// Timestamps come from the telemetry recorder (worker-side steady_clock), so
// TTFT / ITL / migrate stall do not depend on how fast this driver polls.
//
// Outputs in --out-dir:
//   summary.csv        one row per run (config + metrics + counters)
//   requests.csv       one row per request
//   migrations.csv     one row per migration with phase breakdown + stall
//   setup.csv          load_model time per (model, backend, workers, max_seq)
//   trace_<run>.csv    raw trace events (disable with --trace 0)
//   meta.json          run metadata
//
// Needs >= (workers + 3) hardware threads for clean numbers: worker threads,
// the scheduler event thread, the driver, and the migrate driver all run.

#include "bench_common.h"

#include "core/device.h"
#include "core/status.h"
#include "runtime/cluster_config.h"
#include "runtime/cluster_scheduler.h"
#include "runtime/policy/placement_policy.h"
#include "runtime/policy/rebalance_policy.h"
#include "telemetry/log.h"
#include "telemetry/recorder.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <thread>
#include <unordered_map>

namespace {

using bench::Args;
using telemetry::TraceKind;

// ---- policies local to the benchmark ---------------------------------------

// Always places on a fixed worker (OOM if it is full). Creates the skew that
// the rebalance policy is supposed to repair.
class PinnedPlacementPolicy : public PlacementPolicy {
public:
  explicit PinnedPlacementPolicy(size_t idx) : idx_(idx) {}
  StatusOr<size_t> place(std::span<const int32_t>, const GenerateParams&,
                         std::span<const WorkerStat> views) override {
    if (idx_ >= views.size())
      return Status::RuntimeError("pinned worker index out of range");
    if (views[idx_].capacity - views[idx_].inflight <= 0)
      return Status::OOM("pinned worker has no free cache slots");
    return idx_;
  }

private:
  size_t idx_;
};

class NoRebalancePolicy : public RebalancePolicy {
public:
  std::optional<RebalanceDecision> check(std::span<const WorkerStat>,
                                         std::span<const SessionView>) override {
    return std::nullopt;
  }
};

// ---- csv -------------------------------------------------------------------

std::string fnum(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.3f", v);
  return buf;
}
std::string inum(int64_t v) { return std::to_string(v); }

class CsvWriter {
public:
  CsvWriter(const std::string& path, const std::vector<std::string>& cols)
      : f_(path), ncols_(cols.size()) {
    for (size_t i = 0; i < cols.size(); ++i)
      f_ << cols[i] << (i + 1 < cols.size() ? "," : "\n");
  }
  void row(const std::vector<std::string>& cells) {
    if (cells.size() != ncols_)
      std::fprintf(stderr, "csv row has %zu cells, expected %zu\n", cells.size(), ncols_);
    for (size_t i = 0; i < cells.size(); ++i) {
      std::string c = cells[i];
      std::replace(c.begin(), c.end(), ',', ';');
      f_ << c << (i + 1 < cells.size() ? "," : "\n");
    }
    f_.flush();
  }

private:
  std::ofstream f_;
  size_t ncols_;
};

// ---- trace analysis --------------------------------------------------------

struct MigTrace {
  int64_t requested = -1, begin = -1, commit = -1;
  int src = -1, dst = -1;
  int64_t post_t[2] = {-1, -1};  // [0]=send [1]=recv
  int64_t done_t[2] = {-1, -1};
  int64_t bytes = 0;
  bool xfer_ok = true;
};

struct ReqTrace {
  uint64_t id = 0;
  int tag = 0;
  int64_t submit = -1, prefill_start = -1, prefill_end = -1, done = -1;
  int64_t prompt = 0, max_new = 0;
  int first_worker = -1, last_worker = -1;
  bool failed = false;
  std::vector<int64_t> tok_t; // time of every emitted token
  std::vector<MigTrace> migs;
};

using ReqMap = std::unordered_map<uint64_t, ReqTrace>;

ReqMap build_reqs(const std::vector<telemetry::TraceEvent>& events,
                  const std::unordered_map<uint64_t, int>& tags) {
  ReqMap reqs;
  for (const auto& e : events) {
    if (e.kind == TraceKind::Marker || e.kind == TraceKind::RebalanceDecision)
      continue;
    ReqTrace& r = reqs[e.req_id];
    r.id = e.req_id;
    if (auto it = tags.find(e.req_id); it != tags.end())
      r.tag = it->second;
    switch (e.kind) {
    case TraceKind::Submit:
      r.submit = e.t_ns;
      r.prompt = e.a;
      r.max_new = e.b;
      r.first_worker = e.worker;
      r.last_worker = e.worker;
      break;
    case TraceKind::PrefillStart:
      r.prefill_start = e.t_ns;
      break;
    case TraceKind::PrefillEnd:
      if (e.a == 1) {
        r.prefill_end = e.t_ns;
        r.tok_t.push_back(e.t_ns);
        r.last_worker = e.worker;
      }
      break;
    case TraceKind::DecodeEnd:
      if (e.b == 1) {
        r.tok_t.push_back(e.t_ns);
        r.last_worker = e.worker;
      }
      break;
    case TraceKind::MigrateRequested: {
      MigTrace m;
      m.requested = e.t_ns;
      m.src = e.worker;
      m.dst = static_cast<int>(e.a);
      r.migs.push_back(m);
      break;
    }
    case TraceKind::MigrateBegin:
      if (!r.migs.empty())
        r.migs.back().begin = e.t_ns;
      break;
    case TraceKind::MigrateXferPosted:
      if (!r.migs.empty()) {
        MigTrace& m = r.migs.back();
        const int role = static_cast<int>(e.a) ? 1 : 0;
        if (m.post_t[role] < 0 || e.t_ns < m.post_t[role])
          m.post_t[role] = e.t_ns;
        if (m.bytes == 0)
          m.bytes = e.b;
      }
      break;
    case TraceKind::MigrateXferDone:
      if (!r.migs.empty()) {
        MigTrace& m = r.migs.back();
        const int role = static_cast<int>(e.a) ? 1 : 0;
        m.done_t[role] = std::max(m.done_t[role], e.t_ns);
        if (e.b == 0)
          m.xfer_ok = false;
      }
      break;
    case TraceKind::MigrateCommit:
      if (!r.migs.empty())
        r.migs.back().commit = e.t_ns;
      r.last_worker = e.worker;
      break;
    case TraceKind::Done:
      r.done = e.t_ns;
      break;
    case TraceKind::Failed:
      r.failed = true;
      r.done = e.t_ns;
      break;
    default:
      break;
    }
  }
  return reqs;
}

bool overlaps(int64_t a0, int64_t a1, int64_t b0, int64_t b1) { return a0 < b1 && a1 > b0; }

struct ReqStats {
  double ttft_us = 0;
  double e2e_us = 0;
  std::vector<double> itl_all;    // us
  std::vector<double> itl_steady; // us, excludes intervals that contain a migration window
  double baseline_itl_us = 0;
};

ReqStats req_stats(const ReqTrace& r) {
  ReqStats s;
  if (r.submit >= 0 && r.prefill_end >= 0)
    s.ttft_us = static_cast<double>(r.prefill_end - r.submit) / 1e3;
  const int64_t end = !r.tok_t.empty() ? std::max(r.done, r.tok_t.back()) : r.done;
  if (r.submit >= 0 && end >= 0)
    s.e2e_us = static_cast<double>(end - r.submit) / 1e3;
  for (size_t i = 1; i < r.tok_t.size(); ++i) {
    const int64_t t0 = r.tok_t[i - 1], t1 = r.tok_t[i];
    const double us = static_cast<double>(t1 - t0) / 1e3;
    s.itl_all.push_back(us);
    bool in_mig = false;
    for (const MigTrace& m : r.migs) {
      if (m.begin >= 0 && m.commit >= 0 && overlaps(t0, t1, m.begin, m.commit)) {
        in_mig = true;
        break;
      }
    }
    if (!in_mig)
      s.itl_steady.push_back(us);
  }
  s.baseline_itl_us = bench::percentile(s.itl_steady, 50);
  return s;
}

// ---- run configuration / output sinks --------------------------------------

struct RunCfg {
  std::string scenario, model, device, backend;
  std::string config = "na", load = "na", placement = "na", rebalance = "na";
  int workers = 1, slots = 8, max_seq = 2048, sessions = 1, prompt_len = 0, gen_len = 0;
  int threshold = 0, bg = 0, migrations = 0, migrate_at = 0, migrate_every = 0;
  double load_model_ms = 0;
};

struct Sinks {
  std::string dir;
  bool trace = true;
  int run_counter = 0;
  std::unique_ptr<CsvWriter> summary, requests, migrations, setup;
};

enum Tag { kPrimary = 0, kBackground = 1, kWorkload = 2 };

struct SessionSpec {
  int prompt_len = 0;
  int gen_len = 0;
  int64_t arrival_us = 0;
  int tag = kWorkload;
  uint32_t seed = 0;
};

std::vector<int32_t> make_prompt(int len, uint32_t seed) {
  std::vector<int32_t> t(static_cast<size_t>(len));
  uint64_t x = 0x9E3779B97F4A7C15ULL ^ seed;
  for (auto& v : t) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    v = 1 + static_cast<int32_t>((x >> 33) % 1000);
  }
  return t;
}

constexpr int32_t kNeverEos = 1 << 30;

StatusOr<uint64_t> submit_retry(ClusterScheduler& s, const SessionSpec& spec, int* retries) {
  GenerateParams p{.max_new_tokens = spec.gen_len, .eos_id = kNeverEos};
  const int64_t deadline = telemetry::steady_now_ns() + 3'000'000'000LL;
  for (;;) {
    StatusOr<uint64_t> id = s.submit(make_prompt(spec.prompt_len, spec.seed), p);
    if (id.ok() || id.status().code() != Code::kOOM)
      return id;
    if (telemetry::steady_now_ns() > deadline)
      return id;
    ++*retries; // lingering releases from the previous run may still be draining
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

bool wait_all_done(ClusterScheduler& s, const std::vector<uint64_t>& ids, double timeout_s) {
  const int64_t deadline = telemetry::steady_now_ns() + static_cast<int64_t>(timeout_s * 1e9);
  std::vector<bool> done(ids.size(), false);
  size_t remaining = ids.size();
  while (remaining > 0) {
    for (size_t i = 0; i < ids.size(); ++i) {
      if (!done[i] && s.poll(ids[i]).done) {
        done[i] = true;
        --remaining;
      }
    }
    if (remaining == 0)
      break;
    if (telemetry::steady_now_ns() > deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return true;
}

// ---- migrate driver (scenario B) -------------------------------------------

struct MigPlan {
  int first_at = 8;
  int every = 8;
  int count = 3;
};

void migrate_driver(ClusterScheduler* s, uint64_t id, int n_workers, MigPlan plan) {
  size_t home = s->poll(id).worker_idx;
  int next_at = plan.first_at;
  for (int m = 0; m < plan.count; ++m) {
    Response r;
    for (;;) {
      r = s->poll(id);
      if (r.done)
        return;
      if (r.gen_seq_len >= next_at && r.worker_idx == home)
        break;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    const size_t dst = (home + 1) % static_cast<size_t>(n_workers);
    bool requested = false;
    for (int attempt = 0; attempt < 200 && !requested; ++attempt) {
      requested = s->migrate(id, dst).ok();
      if (!requested)
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    if (!requested)
      return;
    const int64_t deadline = telemetry::steady_now_ns() + 30'000'000'000LL;
    for (;;) {
      r = s->poll(id);
      if (r.done)
        return;
      if (r.worker_idx == dst)
        break;
      if (telemetry::steady_now_ns() > deadline)
        return;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    home = dst;
    next_at = r.gen_seq_len + plan.every;
  }
}

// ---- executing and reporting one run ---------------------------------------

struct RunResult {
  bool ok = false;
};

RunResult execute_run(Sinks& out, const RunCfg& cfg, ClusterScheduler& sched,
                      const std::vector<SessionSpec>& specs, const MigPlan* mig, int repeat,
                      bool warmup, double timeout_s) {
  RunResult res;
  telemetry::Recorder& rec = telemetry::Recorder::instance();
  rec.reset();
  rec.enable(true);

  const std::string run_id = cfg.scenario + "_" + std::to_string(out.run_counter++);
  std::vector<uint64_t> ids;
  std::unordered_map<uint64_t, int> tags;
  int retries = 0;
  int rejected = 0;
  std::thread driver;

  const int64_t t0 = telemetry::steady_now_ns();
  for (const SessionSpec& spec : specs) {
    const int64_t due = t0 + spec.arrival_us * 1000;
    while (telemetry::steady_now_ns() < due)
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    StatusOr<uint64_t> id = submit_retry(sched, spec, &retries);
    if (!id.ok()) {
      ++rejected;
      std::fprintf(stderr, "[%s] submit rejected: %s\n", run_id.c_str(),
                   id.status().message().c_str());
      continue;
    }
    ids.push_back(id.value());
    tags[id.value()] = spec.tag;
    if (spec.tag == kPrimary && mig && !driver.joinable())
      driver = std::thread(migrate_driver, &sched, id.value(), cfg.workers, *mig);
  }

  const bool all_done = wait_all_done(sched, ids, timeout_s);
  if (driver.joinable())
    driver.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(30)); // let release events drain
  rec.enable(false);

  if (out.trace)
    (void)rec.dump_csv(out.dir + "/trace_" + run_id + ".csv");

  const std::vector<telemetry::TraceEvent> events = rec.snapshot();
  ReqMap reqs = build_reqs(events, tags);

  // ---- aggregate ----
  std::vector<double> ttft, itl_all, itl_steady;
  int64_t tokens = 0, failed = 0;
  int64_t first_submit = INT64_MAX, last_end = 0;
  std::vector<std::pair<int64_t, int64_t>> windows;
  for (auto& [id, r] : reqs) {
    if (r.submit < 0)
      continue; // late event for a request from a previous run
    ReqStats s = req_stats(r);
    if (r.failed)
      ++failed;
    tokens += static_cast<int64_t>(r.tok_t.size());
    first_submit = std::min(first_submit, r.submit);
    if (!r.tok_t.empty())
      last_end = std::max(last_end, r.tok_t.back());
    if (r.done >= 0)
      last_end = std::max(last_end, r.done);
    if (r.prefill_end >= 0)
      ttft.push_back(s.ttft_us);
    itl_all.insert(itl_all.end(), s.itl_all.begin(), s.itl_all.end());
    itl_steady.insert(itl_steady.end(), s.itl_steady.begin(), s.itl_steady.end());
    for (const MigTrace& m : r.migs)
      if (m.begin >= 0 && m.commit >= 0)
        windows.emplace_back(m.begin, m.commit);
  }
  const double makespan_s =
      first_submit == INT64_MAX ? 0.0 : static_cast<double>(last_end - first_submit) / 1e9;

  // ---- per migration rows ----
  std::vector<double> stall_excess, xfer_us_all, bytes_all;
  int64_t slot_bytes = 0;
  int n_migs = 0;
  for (auto& [id, r] : reqs) {
    if (r.submit < 0)
      continue;
    ReqStats s = req_stats(r);
    for (const MigTrace& m : r.migs) {
      const bool committed = m.begin >= 0 && m.commit >= 0 && m.xfer_ok;
      // tokens emitted when the migration began, and the interval that spans it
      size_t idx = 0;
      if (m.begin >= 0)
        idx = static_cast<size_t>(std::upper_bound(r.tok_t.begin(), r.tok_t.end(), m.begin) -
                                  r.tok_t.begin());
      double gap_us = 0, resume_us = 0;
      if (committed && idx > 0 && idx < r.tok_t.size()) {
        gap_us = static_cast<double>(r.tok_t[idx] - r.tok_t[idx - 1]) / 1e3;
        resume_us = static_cast<double>(r.tok_t[idx] - m.commit) / 1e3;
      }
      const double wait_us = (m.begin >= 0 && m.requested >= 0)
                                 ? static_cast<double>(m.begin - m.requested) / 1e3
                                 : 0.0;
      double xfer_us = 0;
      if (m.done_t[1] >= 0) {
        const int64_t p0 = (m.post_t[0] >= 0 && m.post_t[1] >= 0)
                               ? std::min(m.post_t[0], m.post_t[1])
                               : std::max(m.post_t[0], m.post_t[1]);
        if (p0 >= 0)
          xfer_us = static_cast<double>(m.done_t[1] - p0) / 1e3;
      }
      const double ack_us = (m.commit >= 0 && m.done_t[1] >= 0)
                                ? static_cast<double>(m.commit - m.done_t[1]) / 1e3
                                : 0.0;
      const double total_us = (m.commit >= 0 && m.requested >= 0)
                                  ? static_cast<double>(m.commit - m.requested) / 1e3
                                  : 0.0;
      const double base = s.baseline_itl_us > 0 ? s.baseline_itl_us : 0.0;
      const double excess = committed ? gap_us - base : 0.0;
      const int64_t remaining = r.max_new - static_cast<int64_t>(idx);
      if (committed) {
        ++n_migs;
        stall_excess.push_back(excess);
        xfer_us_all.push_back(xfer_us);
        slot_bytes = std::max(slot_bytes, m.bytes);
      }
      out.migrations->row({run_id,
                           inum(repeat),
                           inum(warmup),
                           cfg.scenario,
                           cfg.model,
                           cfg.backend,
                           cfg.config,
                           inum(cfg.max_seq),
                           cfg.load,
                           inum(static_cast<int64_t>(id)),
                           inum(m.src),
                           inum(m.dst),
                           inum(m.bytes),
                           inum(static_cast<int64_t>(idx)),
                           inum(remaining),
                           fnum(wait_us),
                           fnum(xfer_us),
                           fnum(ack_us),
                           fnum(total_us),
                           fnum(gap_us),
                           fnum(base),
                           fnum(excess),
                           fnum(resume_us),
                           inum(committed ? 1 : 0)});
    }
  }

  // ---- background interference (scenario B) ----
  std::vector<double> bg_during, bg_outside;
  if (!windows.empty()) {
    for (auto& [id, r] : reqs) {
      if (r.tag != kBackground)
        continue;
      for (size_t i = 1; i < r.tok_t.size(); ++i) {
        const int64_t a = r.tok_t[i - 1], b = r.tok_t[i];
        bool during = false;
        for (auto& w : windows)
          if (overlaps(a, b, w.first, w.second)) {
            during = true;
            break;
          }
        (during ? bg_during : bg_outside).push_back(static_cast<double>(b - a) / 1e3);
      }
    }
  }

  // ---- request rows ----
  for (auto& [id, r] : reqs) {
    if (r.submit < 0)
      continue;
    ReqStats s = req_stats(r);
    out.requests->row({run_id, inum(repeat), inum(warmup), cfg.scenario, cfg.backend,
                       inum(static_cast<int64_t>(id)), inum(r.tag), inum(r.prompt),
                       inum(r.max_new), inum(static_cast<int64_t>(r.tok_t.size())),
                       inum(r.first_worker), inum(r.last_worker), fnum(s.ttft_us),
                       fnum(bench::percentile(s.itl_all, 50)),
                       fnum(bench::percentile(s.itl_all, 95)),
                       fnum(s.itl_all.empty() ? 0.0
                                              : *std::max_element(s.itl_all.begin(), s.itl_all.end())),
                       fnum(s.e2e_us), inum(r.failed ? 1 : 0), inum(static_cast<int64_t>(r.migs.size()))});
  }

  // ---- summary row ----
  const auto cnt = rec.counters.snapshot();
  auto counter = [&](const char* name) -> uint64_t {
    for (auto& kv : cnt)
      if (kv.first == name)
        return kv.second;
    return 0;
  };
  out.summary->row(
      {run_id,
       inum(repeat),
       inum(warmup),
       cfg.scenario,
       cfg.model,
       cfg.device,
       cfg.backend,
       cfg.config,
       cfg.placement,
       cfg.rebalance,
       cfg.load,
       inum(cfg.workers),
       inum(cfg.slots),
       inum(cfg.max_seq),
       inum(cfg.sessions),
       inum(cfg.prompt_len),
       inum(cfg.gen_len),
       inum(cfg.threshold),
       inum(cfg.bg),
       fnum(cfg.load_model_ms),
       inum(static_cast<int64_t>(ids.size())),
       inum(rejected),
       inum(failed),
       inum(all_done ? 0 : 1),
       inum(tokens),
       fnum(makespan_s),
       fnum(makespan_s > 0 ? static_cast<double>(tokens) / makespan_s : 0.0),
       fnum(bench::mean(ttft)),
       fnum(bench::percentile(ttft, 50)),
       fnum(bench::percentile(ttft, 95)),
       fnum(bench::percentile(itl_all, 50)),
       fnum(bench::percentile(itl_all, 95)),
       fnum(bench::percentile(itl_all, 99)),
       fnum(itl_all.empty() ? 0.0 : *std::max_element(itl_all.begin(), itl_all.end())),
       fnum(bench::percentile(itl_steady, 50)),
       inum(n_migs),
       inum(slot_bytes),
       fnum(bench::percentile(xfer_us_all, 50)),
       fnum(bench::percentile(stall_excess, 50)),
       fnum(bench::percentile(bg_during, 50)),
       fnum(bench::percentile(bg_outside, 50)),
       inum(static_cast<int64_t>(bg_during.size())),
       inum(static_cast<int64_t>(counter("migrates_started"))),
       inum(static_cast<int64_t>(counter("migrates_completed"))),
       inum(static_cast<int64_t>(counter("migrates_failed"))),
       inum(static_cast<int64_t>(counter("bytes_migrated"))),
       inum(static_cast<int64_t>(counter("rebalance_checks"))),
       inum(static_cast<int64_t>(counter("rebalance_decisions"))),
       inum(static_cast<int64_t>(counter("slot_rejects"))),
       inum(static_cast<int64_t>(counter("kv_overflows"))),
       inum(retries)});

  std::fprintf(stderr,
               "[%s]%s %s/%s w=%d sess=%zu tok/s=%.1f ttft_p50=%.0fus itl_p50=%.0fus "
               "itl_p99=%.0fus migs=%d%s\n",
               run_id.c_str(), warmup ? " (warmup)" : "", cfg.model.c_str(), cfg.backend.c_str(),
               cfg.workers, ids.size(), makespan_s > 0 ? static_cast<double>(tokens) / makespan_s : 0.0,
               bench::percentile(ttft, 50), bench::percentile(itl_all, 50),
               bench::percentile(itl_all, 99), n_migs, all_done ? "" : "  TIMEOUT");

  res.ok = all_done && failed == 0 && rejected == 0;
  return res;
}

// ---- model load helper -----------------------------------------------------

bool load_cluster(ClusterScheduler& sched, RunCfg& cfg, const std::string& model_dir, Device dev,
                  TransportBackend backend, Sinks& out) {
  ClusterConfig cc;
  cc.model_dir = model_dir;
  cc.device = dev;
  cc.n_workers = cfg.workers;
  cc.num_slots = cfg.slots;
  cc.transport = backend;
  cc.max_seq_length = cfg.max_seq;
  const int64_t t0 = telemetry::steady_now_ns();
  Status s = sched.load_model(cc);
  cfg.load_model_ms = static_cast<double>(telemetry::steady_now_ns() - t0) / 1e6;
  if (!s.ok()) {
    std::fprintf(stderr, "load_model failed (%s, %s, w=%d, max_seq=%d): %s\n", cfg.model.c_str(),
                 cfg.backend.c_str(), cfg.workers, cfg.max_seq, s.message().c_str());
    return false;
  }
  out.setup->row({cfg.scenario, cfg.model, cfg.device, cfg.backend, inum(cfg.workers),
                  inum(cfg.slots), inum(cfg.max_seq), fnum(cfg.load_model_ms)});
  return true;
}

struct Common {
  std::vector<std::string> model_dirs;
  Device device = Device::GPU;
  std::vector<std::string> backends;
  int slots = 8;
  int prompt_len = 128;
  int gen_len = 64;
  int repeats = 5;
  int warmup_runs = 1;
  double timeout_s = 600;
  uint32_t seed = 1;
};

std::string model_label(const std::string& dir) {
  std::string d = dir;
  while (!d.empty() && d.back() == '/')
    d.pop_back();
  const auto p = d.find_last_of('/');
  return p == std::string::npos ? d : d.substr(p + 1);
}

// Runs warmup + repeats of the same workload.
void run_repeats(Sinks& out, const RunCfg& cfg, ClusterScheduler& sched,
                 const std::vector<SessionSpec>& specs, const MigPlan* mig, const Common& c) {
  for (int w = 0; w < c.warmup_runs; ++w) {
    execute_run(out, cfg, sched, specs, mig, /*repeat=*/-1, /*warmup=*/true, c.timeout_s);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  for (int rep = 0; rep < c.repeats; ++rep) {
    execute_run(out, cfg, sched, specs, mig, rep, /*warmup=*/false, c.timeout_s);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

// ---- scenario A: replica scaling ------------------------------------------

void scenario_scale(Sinks& out, const Args& args, const Common& c) {
  const auto worker_list = args.i64_list("workers", {1, 2});
  const auto session_list = args.i64_list("sessions", {1, 2, 4, 8, 16});
  const int max_sessions = static_cast<int>(*std::max_element(session_list.begin(), session_list.end()));
  const int max_seq = static_cast<int>(args.i64("max-seq", c.prompt_len + c.gen_len + 16));
  // Scale ignores the transport; default to the cheapest backend.
  const auto backends = args.has("backends") ? c.backends : std::vector<std::string>{"memcpy"};

  for (const std::string& dir : c.model_dirs) {
    for (int64_t workers : worker_list) {
      for (const std::string& bname : backends) {
        auto backend = bench::parse_backend(bname);
        if (!backend)
          continue;
        RunCfg cfg;
        cfg.scenario = "scale";
        cfg.model = model_label(dir);
        cfg.device = bench::device_name(c.device);
        cfg.backend = bname;
        cfg.workers = static_cast<int>(workers);
        cfg.slots = std::max(c.slots, max_sessions); // one worker must be able to hold all
        cfg.max_seq = max_seq;
        cfg.prompt_len = c.prompt_len;
        cfg.gen_len = c.gen_len;

        // Automatic rebalancing would add migrations the scenario did not ask for.
        ClusterScheduler sched(nullptr, std::make_unique<NoRebalancePolicy>());
        if (!load_cluster(sched, cfg, dir, c.device, *backend, out))
          continue;
        for (int64_t n : session_list) {
          RunCfg run = cfg;
          run.sessions = static_cast<int>(n);
          std::vector<SessionSpec> specs;
          for (int i = 0; i < n; ++i)
            specs.push_back(SessionSpec{c.prompt_len, c.gen_len, 0, kWorkload,
                                        c.seed + static_cast<uint32_t>(i)});
          run_repeats(out, run, sched, specs, nullptr, c);
        }
      }
    }
  }
}

// ---- scenario B: migration cost -------------------------------------------

void scenario_migrate(Sinks& out, const Args& args, const Common& c) {
  const auto max_seqs = args.i64_list("max-seq", {2048});
  const auto loads = args.str_list("loads", {"idle", "busy"});
  const int bg_sessions = static_cast<int>(args.i64("bg-sessions", 2));
  MigPlan plan;
  plan.first_at = static_cast<int>(args.i64("migrate-at", 8));
  plan.every = static_cast<int>(args.i64("migrate-every", 8));
  plan.count = static_cast<int>(args.i64("migrations", 3));
  const int gen_len = std::max(c.gen_len, plan.first_at + plan.count * (plan.every + 4) + 4);

  for (const std::string& dir : c.model_dirs) {
    for (const std::string& bname : c.backends) {
      auto backend = bench::parse_backend(bname);
      if (!backend) {
        std::fprintf(stderr, "unknown backend '%s'\n", bname.c_str());
        continue;
      }
      for (int64_t max_seq : max_seqs) {
        if (c.prompt_len + gen_len + 1 > max_seq) {
          std::fprintf(stderr, "skip max_seq=%lld: prompt+gen exceeds it\n",
                       static_cast<long long>(max_seq));
          continue;
        }
        RunCfg cfg;
        cfg.scenario = "migrate";
        cfg.model = model_label(dir);
        cfg.device = bench::device_name(c.device);
        cfg.backend = bname;
        cfg.workers = 2;
        cfg.slots = std::max(c.slots, bg_sessions + 2);
        cfg.max_seq = static_cast<int>(max_seq);
        cfg.prompt_len = c.prompt_len;
        cfg.gen_len = gen_len;
        cfg.migrations = plan.count;
        cfg.migrate_at = plan.first_at;
        cfg.migrate_every = plan.every;

        // Automatic rebalancing would add migrations the scenario did not ask for.
        ClusterScheduler sched(nullptr, std::make_unique<NoRebalancePolicy>());
        if (!load_cluster(sched, cfg, dir, c.device, *backend, out))
          continue;
        for (const std::string& load : loads) {
          RunCfg run = cfg;
          run.load = load;
          run.bg = (load == "busy") ? bg_sessions : 0;
          std::vector<SessionSpec> specs;
          specs.push_back(SessionSpec{c.prompt_len, gen_len, 0, kPrimary, c.seed});
          for (int i = 0; i < run.bg; ++i)
            specs.push_back(SessionSpec{c.prompt_len, gen_len, 0, kBackground,
                                        c.seed + 100 + static_cast<uint32_t>(i)});
          run.sessions = static_cast<int>(specs.size());
          run_repeats(out, run, sched, specs, &plan, c);
        }
      }
    }
  }
}

// ---- scenario C: rebalancing ----------------------------------------------

void scenario_rebalance(Sinks& out, const Args& args, const Common& c) {
  const int sessions = static_cast<int>(args.i64("sessions-rebalance", 8));
  const auto thresholds = args.i64_list("thresholds", {2, 4});
  const auto configs = args.str_list("configs", {"pinned_off", "pinned_on", "balanced_off"});
  const auto max_seqs = args.i64_list("max-seq", {2048});
  const double arrival_ms = args.f64("arrival-ms", 0.0);
  const int long_mult = 4;

  // Long-tailed generation lengths: 50% short, 40% medium, 10% long.
  std::mt19937 rng(c.seed);
  std::vector<int> gens;
  for (int i = 0; i < sessions; ++i) {
    const double u = std::uniform_real_distribution<double>(0, 1)(rng);
    gens.push_back(u < 0.5 ? std::max(4, c.gen_len / 4) : (u < 0.9 ? c.gen_len : c.gen_len * long_mult));
  }

  for (const std::string& dir : c.model_dirs) {
    for (const std::string& bname : c.backends) {
      auto backend = bench::parse_backend(bname);
      if (!backend) {
        std::fprintf(stderr, "unknown backend '%s'\n", bname.c_str());
        continue;
      }
      for (int64_t max_seq : max_seqs) {
        if (c.prompt_len + c.gen_len * long_mult + 1 > max_seq) {
          std::fprintf(stderr, "skip max_seq=%lld: prompt+4*gen exceeds it\n",
                       static_cast<long long>(max_seq));
          continue;
        }
        for (const std::string& conf : configs) {
          const bool pinned = conf.rfind("pinned", 0) == 0;
          const bool rebalance_on = conf.size() > 3 && conf.substr(conf.size() - 3) == "_on";
          const std::vector<int64_t> ths = rebalance_on ? thresholds : std::vector<int64_t>{0};
          for (int64_t th : ths) {
            RunCfg cfg;
            cfg.scenario = "rebalance";
            cfg.model = model_label(dir);
            cfg.device = bench::device_name(c.device);
            cfg.backend = bname;
            cfg.config = conf;
            cfg.placement = pinned ? "pinned0" : "least_loaded";
            cfg.rebalance = rebalance_on ? "on" : "off";
            cfg.threshold = static_cast<int>(th);
            cfg.workers = 2;
            cfg.slots = std::max(c.slots, sessions);
            cfg.max_seq = static_cast<int>(max_seq);
            cfg.sessions = sessions;
            cfg.prompt_len = c.prompt_len;
            cfg.gen_len = c.gen_len;

            std::unique_ptr<PlacementPolicy> place;
            if (pinned)
              place = std::make_unique<PinnedPlacementPolicy>(0);
            std::unique_ptr<RebalancePolicy> reb;
            if (rebalance_on)
              reb = std::make_unique<SimpleRebalancePolicy>(static_cast<int>(th));
            else
              reb = std::make_unique<NoRebalancePolicy>();

            ClusterScheduler sched(std::move(place), std::move(reb));
            if (!load_cluster(sched, cfg, dir, c.device, *backend, out))
              continue;

            std::vector<SessionSpec> specs;
            for (int i = 0; i < sessions; ++i)
              specs.push_back(SessionSpec{c.prompt_len, gens[static_cast<size_t>(i)],
                                          static_cast<int64_t>(arrival_ms * 1000.0 * i), kWorkload,
                                          c.seed + static_cast<uint32_t>(i)});
            run_repeats(out, cfg, sched, specs, nullptr, c);
          }
        }
      }
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  Args args(argc, argv);
  if (args.has("help") || argc == 1) {
    std::puts(
        "cluster_bench --scenario scale|migrate|rebalance|all\n"
        "  --model-dirs models/falcon-h1-0.5b-base[,models/mamba2-130m-hf]\n"
        "  --device GPU|CPU --backends memcpy,nccl,nixl\n"
        "  --prompt-len 128 --gen-len 64 --slots 8 --repeats 5 --warmup-runs 1\n"
        "  --out-dir DIR --trace 1 --timeout-s 600 --seed 1\n"
        "scale:     --workers 1,2 --sessions 1,2,4,8,16 [--max-seq N]\n"
        "migrate:   --max-seq 512,1024,2048 --loads idle,busy --bg-sessions 2\n"
        "           --migrate-at 8 --migrate-every 8 --migrations 3\n"
        "rebalance: --sessions-rebalance 8 --configs pinned_off,pinned_on,balanced_off\n"
        "           --thresholds 2,4 --max-seq 2048 --arrival-ms 0\n"
        "Env: MAMBASERVE_LOG=debug|info|warn|error|off");
    return 0;
  }

  Common c;
  c.model_dirs = args.str_list("model-dirs", {args.str("model-dir", "models/falcon-h1-0.5b-base")});
  c.device = bench::parse_device(args.str("device", "GPU")).value_or(Device::GPU);
  c.backends = args.str_list("backends", {"memcpy"});
  c.slots = static_cast<int>(args.i64("slots", 8));
  c.prompt_len = static_cast<int>(args.i64("prompt-len", 128));
  c.gen_len = static_cast<int>(args.i64("gen-len", 64));
  c.repeats = static_cast<int>(args.i64("repeats", 5));
  c.warmup_runs = static_cast<int>(args.i64("warmup-runs", 1));
  c.timeout_s = args.f64("timeout-s", 600);
  c.seed = static_cast<uint32_t>(args.i64("seed", 1));

  Sinks out;
  out.dir = args.str("out-dir", "results/cluster_" + bench::timestamp_str());
  out.trace = args.i64("trace", 1) != 0;
  bench::make_dirs(out.dir);

  out.summary = std::make_unique<CsvWriter>(
      out.dir + "/summary.csv",
      std::vector<std::string>{
          "run_id", "repeat", "warmup", "scenario", "model", "device", "backend", "config",
          "placement", "rebalance", "load", "workers", "slots", "max_seq", "sessions",
          "prompt_len", "gen_len", "threshold", "bg_sessions", "load_model_ms", "requests",
          "rejected", "failed", "timed_out", "tokens", "makespan_s", "throughput_tok_s",
          "ttft_mean_us", "ttft_p50_us", "ttft_p95_us", "itl_p50_us", "itl_p95_us", "itl_p99_us",
          "itl_max_us", "itl_steady_p50_us", "migrations", "slot_bytes", "mig_xfer_p50_us",
          "mig_stall_excess_p50_us", "bg_itl_during_p50_us", "bg_itl_outside_p50_us",
          "bg_itl_during_n", "ctr_migrates_started", "ctr_migrates_completed",
          "ctr_migrates_failed", "ctr_bytes_migrated", "ctr_rebalance_checks",
          "ctr_rebalance_decisions", "ctr_slot_rejects", "ctr_kv_overflows", "submit_retries"});
  out.requests = std::make_unique<CsvWriter>(
      out.dir + "/requests.csv",
      std::vector<std::string>{"run_id", "repeat", "warmup", "scenario", "backend", "req_id", "tag",
                               "prompt_len", "gen_requested", "tokens", "first_worker",
                               "last_worker", "ttft_us", "itl_p50_us", "itl_p95_us", "itl_max_us",
                               "e2e_us", "failed", "migrations"});
  out.migrations = std::make_unique<CsvWriter>(
      out.dir + "/migrations.csv",
      std::vector<std::string>{"run_id", "repeat", "warmup", "scenario", "model", "backend",
                               "config", "max_seq", "load", "req_id", "src", "dst", "bytes",
                               "tokens_before", "tokens_remaining", "wait_boundary_us", "xfer_us",
                               "commit_ack_us", "total_us", "stall_gap_us", "baseline_itl_us",
                               "stall_excess_us", "resume_us", "committed"});
  out.setup = std::make_unique<CsvWriter>(
      out.dir + "/setup.csv",
      std::vector<std::string>{"scenario", "model", "device", "backend", "workers", "slots",
                               "max_seq", "load_model_ms"});

  telemetry::MetaKV meta = bench::common_meta("cluster_bench", args);
  if (Status s = telemetry::write_meta_json(out.dir + "/meta.json", meta); !s.ok())
    std::fprintf(stderr, "warning: %s\n", s.message().c_str());

  std::vector<std::string> scenarios = args.str_list("scenario", {"migrate"});
  if (std::find(scenarios.begin(), scenarios.end(), "all") != scenarios.end())
    scenarios = {"scale", "migrate", "rebalance"};

  for (const std::string& sc : scenarios) {
    if (sc == "scale")
      scenario_scale(out, args, c);
    else if (sc == "migrate")
      scenario_migrate(out, args, c);
    else if (sc == "rebalance")
      scenario_rebalance(out, args, c);
    else
      std::fprintf(stderr, "unknown scenario '%s'\n", sc.c_str());
  }

  std::fprintf(stderr, "results in %s\n", out.dir.c_str());
  return 0;
}
