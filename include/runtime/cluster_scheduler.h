#pragma once

#include "comm/comm_factory.h"
#include "core/status.h"
#include "io/config.h"
#include "proto/worker.pb.h"
#include "runtime/cluster_config.h"
#include "runtime/ipc/ipc_comm.h"
#include "runtime/policy/placement_policy.h"
#include "runtime/policy/rebalance_policy.h"
#include "runtime/session.h"
#include "runtime/worker_slot.h"

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

// takes in a config and already-created workers (see create_cluster_workers in
// worker_factory.h; threads or processes per ClusterConfig::worker_mode), a placement policy
// that returns a worker view for a given request
// sends cmds to each worker via send_cmd_ over the parent end of its IpcChannel
// listens to worker events: ingress_loop_ drains every channel into eventq_,
// event_loop_ applies them to sessions
class ClusterScheduler {
private:
  // event queue that ingress_loop_ pushes worker events into
  std::mutex event_mu_;
  std::condition_variable event_cv_;
  std::deque<mambaserve::Event> eventq_;
  std::thread event_thread_;
  std::thread ingress_thread_;
  bool stop_ = false;

  // Process-mode TraceBatch rendezvous (ingress → collect_worker_traces).
  std::mutex trace_mu_;
  std::condition_variable trace_cv_;
  int pending_trace_replies_ = 0;
  std::vector<bool> trace_reply_seen_;

  // array of workers
  std::vector<WorkerSlot> workers_;
  // array of worker stats (index-aligned with workers_)
  std::vector<WorkerStat> worker_stats_;

  ClusterConfig cfg_{};

  // backend control-plane brain; the scheduler only forwards transport
  // messages and migrate lifecycle hooks to it (see comm/transport_control_plane.h)
  std::unique_ptr<TransportControlPlane> transport_plane_;
  // what the plane may ask of the scheduler: deliver to workers
  struct Sender final : TransportSender {
    explicit Sender(ClusterScheduler& s) : sched(s) {}
    void send_transport(size_t worker_idx, const mambaserve::TransportControl& msg) override;
    void broadcast_transport(const mambaserve::TransportControl& msg) override;
    size_t n_workers() const override;
    ClusterScheduler& sched;
  };
  Sender transport_sender_{*this};

  // policy that decides which worker to be chosen from
  std::unique_ptr<PlacementPolicy> policy_;
  std::unique_ptr<RebalancePolicy> rebalance_;

  // for simple request id generation
  std::atomic<uint64_t> req_id_counter_{0};

  // session_mu_ guards sessions_; submit()/poll() take it from the
  // caller's thread, the event loop takes it while applying events.
  mutable std::mutex session_mu_;
  std::unordered_map<uint64_t, Session> sessions_;

  // drains eventq_ and applies PrefillDone/DecodeDone/Released/Migrated
  // to the matching session, driving the next command (decode/release).
  void send_cmd_(size_t worker_idx, mambaserve::Command cmd);
  void send_trsp_(size_t worker_idx, const mambaserve::TransportControl& trsp);
  void ingress_loop_();
  void event_loop_();
  void handle_event_(mambaserve::Event ev);
  void on_prefill_event_(const mambaserve::PrefillEvent& ev);
  void on_decode_event_(const mambaserve::DecodeEvent& ev);
  void on_release_event_(const mambaserve::ReleaseEvent& ev);
  void on_migrate_event_(const mambaserve::MigrateEvent& ev);
  void begin_migrate_(Session& sess, uint64_t req_id, size_t dst_idx);
  void fail_migrate_(Session& sess, uint64_t req_id, const Status& err);
  void cancel_pending_migrate_(Session& sess);
  // After a successful Prefill/Decode token: Done, begin migrate, or continue decode.
  void continue_after_token_(Session& sess, uint64_t req_id, int32_t token);
  // Call with session_mu_ held.
  Status migrate_locked_(uint64_t req_id, size_t dst_idx);
  void maybe_rebalance_();
  // Drops worker slots (joins threads / reaps processes) and aligned stats.
  void clear_workers_() noexcept;

public:
  // Defaults to SimplePlacementPolicy / SimpleRebalancePolicy when null.
  explicit ClusterScheduler(std::unique_ptr<PlacementPolicy> policy = nullptr,
                            std::unique_ptr<RebalancePolicy> rebalance = nullptr);

  // submit token ids directly; tokenization stays in the host (see runtime/tokenizer.h).
  StatusOr<uint64_t> submit(std::vector<int32_t> tokens, GenerateParams params);
  Response poll(uint64_t req_id) const;

  // Request a migrate to dst_idx. Takes effect at the next decode boundary
  // (after the in-flight decode settles) so token history stays contiguous.
  Status migrate(uint64_t req_id, size_t dst_idx);

  // Takes ownership of `workers` (one per cfg.n_workers, index == device id), waits for
  // each to report Ready, then starts scheduling. Workers are destroyed (joined/reaped)
  // on failure. The host creates them with create_cluster_workers(cfg).
  Status start(const ClusterConfig& cfg, std::vector<WorkerSlot> workers);

  // Process-mode only: tell each worker to clear its local recorder and align its
  // epoch to `epoch_ns` (usually Recorder::instance().epoch_ns() after parent reset).
  // Thread mode is a no-op (workers already share the parent recorder).
  void sync_worker_traces(int64_t epoch_ns);

  // Process-mode only: ask each worker for a TraceBatch, rewrite into the parent
  // epoch, and ingest into Recorder::instance(). Thread mode is a no-op.
  Status collect_worker_traces(double timeout_s = 10.0);

  // shutdown all workers (waits for all requests to be done) then shutsdown
  void shutdown();

  // calls shutdown
  ~ClusterScheduler();
};
