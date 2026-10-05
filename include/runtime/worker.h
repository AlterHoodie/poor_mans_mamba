#pragma once

#include "comm/comm_agent.h"
#include "comm/comm_factory.h"
#include "core/device.h"
#include "core/status.h"
#include "proto/worker.pb.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/cluster_config.h"
#include "runtime/ipc/ipc_comm.h"
#include "runtime/runner/runner.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>

struct GenerateParams {
  int max_new_tokens = 32;
  int eos_id = -1; // required for generate; <0 rejected
};

struct WorkerStat {
  size_t index;
  int capacity;
  int inflight;
};

// Everything a worker needs to build its own device allocator, model replica, cache
// pool and comm agent. Only plain config + the child IPC end cross the parent/worker
// boundary, so the same bootstrap serves a worker thread and a spawned worker process
// (see worker_process.h).
struct WorkerBootstrap {
  size_t index = 0;
  int device_id = -1;
  ClusterConfig cfg;
  std::unique_ptr<IpcChannel> ipc;
};

// Where a Worker's execute loop runs.
//   Thread: the constructor starts a std::thread (thread-mode cluster workers).
//   Inline: nothing starts; the caller drives the loop with run() (process-mode child).
enum class WorkerRunMode { Thread, Inline };

// Worker owns the child end of an IpcChannel and runs the execute loop.
// The loop first initializes (device bind, model load, cache pool, comm agent) and
// reports the outcome in a ReadyEvent; on failure it exits without serving commands.
class Worker {
private:
  size_t index_ = 0;
  int device_id_ = -1;
  ClusterConfig cfg_;
  std::unique_ptr<DeviceAllocator> alloc_;
  std::unique_ptr<CachePool> pool_;
  std::unique_ptr<Runner> runner_;
  std::unique_ptr<CommAgent> comm_agent_;
  std::unique_ptr<IpcChannel> ipc_;

  bool stop_ = false;
  bool init_ok_ = false;
  std::thread thread_;

  struct PendingXfer {
    XferHandle handle;
    XferRole role = XferRole::Send;
  };

  std::unordered_map<uint64_t, CacheHandle> cache_handles_;
  std::unordered_map<uint64_t, PendingXfer> pending_transfers_;

  StatusOr<int32_t> sample_(Tensor& logits);

  // creates cache handle, checks if we can register or not if not returns some status
  Status register_(uint64_t req_id);

  // releases cache handle
  Status release_(uint64_t req_id);

  // prefill, cache the states and return new token
  mambaserve::PrefillEvent prefill_(uint64_t req_id, std::span<const int32_t> tokens);

  // one forward pass, update cache, and return new token
  mambaserve::DecodeEvent decode_(uint64_t req_id, const int32_t token);

  // Posts a transfer. On success stores the handle in pending_transfers_ and
  // returns nullopt (MigrateEvent is emitted later by poll_transfer_states_).
  // On immediate failure returns a MigrateEvent to emit now.
  std::optional<mambaserve::MigrateEvent> migrate_(const mambaserve::MigrateCmd& cmd);
  std::optional<mambaserve::MigrateEvent> migrate_recv_(const mambaserve::MigrateCmd& cmd);
  std::optional<mambaserve::MigrateEvent> migrate_send_(const mambaserve::MigrateCmd& cmd);
  std::optional<mambaserve::MigrateEvent> post_xfer_(const mambaserve::MigrateCmd& cmd, int64_t seq_len);

  // loops through all the pending transfers, when any transfer is successfull or failed
  // it emits the appropriate MigrateEvent to the parent over ipc_
  void poll_transfer_states_();

  // wraps the event in an Envelope and sends it to the parent over ipc_
  // (no-op if ipc_ is null; send failures are logged)
  void emit_event_(mambaserve::Event ev);

  // wraps a transport control message in an Envelope and sends it to the parent
  void emit_transport_(mambaserve::TransportControl msg);

  // applies a transport control message routed from the parent to the comm agent
  void handle_transport_(const mambaserve::TransportControl& msg);

  // executes a single command from the parent and emits the resulting event(s)
  void handle_command_(const mambaserve::Command& cmd);

  // Binds the device and builds alloc_/runner_/pool_/comm_agent_ from cfg_.
  Status init_();

  // loop: receive an Envelope from the parent over ipc_, dispatch commands via
  // handle_command_, and poll pending transfers between messages (non-blocking
  // recv while transfers are pending, blocking otherwise).
  // Exits once stop_ is set and no transfers are pending, or when ipc_ closes.
  void loop_();

public:
  // Thread mode starts the worker thread immediately; model/allocator/pool/agent are
  // built there. Inline mode only stores the bootstrap; call run().
  explicit Worker(WorkerBootstrap boot, WorkerRunMode mode = WorkerRunMode::Thread);

  // Inline mode only: runs init + the execute loop on the calling thread until the
  // parent shuts the worker down or the channel closes. Returns 0 if init succeeded
  // and the loop ended normally, 1 if init failed.
  int run();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  ~Worker();
};
