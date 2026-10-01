#pragma once

#include "comm/comm_factory.h"
#include "core/status.h"
#include "io/config.h"
#include "runtime/cluster_config.h"
#include "runtime/model_registry.h"
#include "runtime/policy/placement_policy.h"
#include "runtime/policy/rebalance_policy.h"
#include "runtime/session.h"
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

  ClusterConfig cfg_{};
  std::shared_ptr<TransportContext> transport_ctx_;

  // policy that decides which worker to be chosen from
  std::unique_ptr<PlacementPolicy> policy_;
  std::unique_ptr<RebalancePolicy> rebalance_;

  // for simple request id generation
  std::atomic<uint64_t> req_id_counter_{0};

  // session_mu_ guards sessions_; submit()/poll() take it from the
  // caller's thread, the event loop takes it while applying events.
  mutable std::mutex session_mu_;
  std::unordered_map<uint64_t, Session> sessions_;

  StatusOr<WorkerSlot> create_worker_(const ModelEntry& entry, int device_id);

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
  // Call with session_mu_ held.
  Status migrate_locked_(uint64_t req_id, size_t dst_idx);
  void maybe_rebalance_();

public:
  // Defaults to SimplePlacementPolicy / SimpleRebalancePolicy when null.
  explicit ClusterScheduler(std::unique_ptr<PlacementPolicy> policy = nullptr,
                            std::unique_ptr<RebalancePolicy> rebalance = nullptr);

  // submit token ids directly; tokenization is not wired into the
  // cluster scheduler yet (see runtime/tokenizer.h for the single-model
  // path used by main.cpp / benchmarks).
  StatusOr<uint64_t> submit(std::vector<int32_t> tokens, GenerateParams params);
  Response poll(uint64_t req_id) const;

  // Request a migrate to dst_idx. Takes effect at the next decode boundary
  // (after the in-flight decode settles) so token history stays contiguous.
  Status migrate(uint64_t req_id, size_t dst_idx);

  Status load_model(const ClusterConfig& cfg);

  // shutdown all workers (waits for all requests to be done) then shutsdown
  void shutdown();

  // calls shutdown
  ~ClusterScheduler();
};
