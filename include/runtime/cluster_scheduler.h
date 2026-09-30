#pragma once

#include "core/status.h"
#include "io/config.h"
#include "runtime/model_registry.h"
#include "runtime/policy/placement_policy.h"
#include "runtime/worker.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// Response is a snapshot of session state at poll() time: tokens generated so
// far, whether generation has finished (Done or Failed), and terminal status.
struct Response {
  uint64_t req_id = 0;
  std::vector<int32_t> tokens; // generated tokens so far
  int gen_seq_len = 0;
  size_t worker_idx = 0;
  bool done = false;
  Status s = Status::Ok();
};

struct WorkerSlot {
  int device_id = -1;
  std::unique_ptr<Worker> worker;
};

enum class SessionPhase { Prefilling, Decoding, Migrating, Done, Failed };

// Control-plane bookkeeping for one in-flight generate() request. Only the
// event-processing thread and poll()/submit() (under session_mu_) touch this.
struct Session {
  std::vector<int32_t> generated_tokens;
  GenerateParams params;
  SessionPhase phase = SessionPhase::Prefilling;
  Status s = Status::Ok();
  size_t worker_idx = 0;
  size_t migrate_dst_idx = 0;
  int migrate_acks = 0;
  // When set, the next decode boundary starts migrate instead of another Decode.
  bool migrate_pending = false;
};

// takes in some sort of config , spins up that many threads of workers
// , a placement policy that returns a worker view for a given request
// pushes cmds using workers->enqueue
// listens to worker events
class ClusterScheduler {
private:
  // for event queue workers pushing events into
  std::mutex event_mu_;
  std::condition_variable event_cv_;
  std::deque<Event> eventq_;
  std::thread event_thread_;
  bool stop_ = false;

  // array of workers
  std::vector<WorkerSlot> workers_;
  // array of worker stats (index-aligned with workers_)
  std::vector<WorkerStat> worker_stats_;

  // policy that decides which worker to be chosen from
  std::unique_ptr<PlacementPolicy> policy_;

  // for simple request id generation
  std::atomic<uint64_t> req_id_counter_{0};

  // session_mu_ guards sessions_; submit()/poll() take it from the
  // caller's thread, the event loop takes it while applying events.
  mutable std::mutex session_mu_;
  std::unordered_map<uint64_t, Session> sessions_;

  StatusOr<WorkerSlot> create_worker_(const ModelEntry& entry, Device kind, int device_id,
                                      int num_slots);

  // drains eventq_ and applies PrefillDone/DecodeDone/Released/Migrated
  // to the matching session, driving the next command (decode/release).
  void event_loop_();
  void handle_event_(Event ev);
  void on_prefill_event_(const PrefillEvent& ev);
  void on_decode_event_(const DecodeEvent& ev);
  void on_release_event_(const ReleaseEvent& ev);
  void on_migrate_event_(const MigrateEvent& ev);
  void begin_migrate_(Session& sess, uint64_t req_id, size_t dst_idx);
  void fail_migrate_(Session& sess, uint64_t req_id, const Status& err);
  void cancel_pending_migrate_(Session& sess);
  // After a successful Prefill/Decode token: Done, begin migrate, or continue decode.
  void continue_after_token_(Session& sess, uint64_t req_id, int32_t token);

public:
  // Defaults to SimplePlacementPolicy when policy is null / default-constructed.
  explicit ClusterScheduler(std::unique_ptr<PlacementPolicy> policy = nullptr);

  // submit token ids directly; tokenization is not wired into the
  // cluster scheduler yet (see runtime/tokenizer.h for the single-model
  // path used by main.cpp / benchmarks).
  StatusOr<uint64_t> submit(std::vector<int32_t> tokens, GenerateParams params);
  Response poll(uint64_t req_id) const;

  // Request a migrate to dst_idx. Takes effect at the next decode boundary
  // (after the in-flight decode settles) so token history stays contiguous.
  Status migrate(uint64_t req_id, size_t dst_idx);

  Status load_model(const std::string model_dir, Device kind, int n_devices, int num_slots = 8);

  // shutdown all workers (waits for all requests to be done) then shutsdown
  void shutdown();

  // calls shutdown
  ~ClusterScheduler();
};
