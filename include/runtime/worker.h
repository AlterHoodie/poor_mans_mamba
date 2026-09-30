#pragma once

#include "comm/comm_agent.h"
#include "core/device.h"
#include "core/status.h"
#include "runtime/cache/cache_pool.h"
#include "runtime/runner/runner.h"

#include <sys/types.h>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

struct GenerateParams {
  int max_new_tokens = 32;
  int eos_id = -1; // required for generate; <0 rejected
};

struct PrefillCmd{
  uint64_t req_id;
  std::vector<int32_t> tokens;
  GenerateParams params;
};

struct DecodeCmd{
  uint64_t req_id;
  int32_t token;
};

struct ReleaseCmd{
  uint64_t req_id;
};

struct MigrateCmd{
  uint64_t req_id = 0;
  XferRole role = XferRole::Send;
  int peer_device_id = -1;
};

struct PrefillEvent{
  uint64_t req_id;
  const int32_t token = -1;
  Status s;
};

struct DecodeEvent{
  uint64_t req_id;
  const int32_t token = -1;
  Status s;
};

struct ReleaseEvent{
  uint64_t req_id;
  size_t worker_idx = 0;
  Status s;
};

struct MigrateEvent{
  uint64_t req_id;
  XferRole role = XferRole::Send;
  Status s;
}; 

using Command = std::variant<PrefillCmd, DecodeCmd, ReleaseCmd, MigrateCmd>;
using Event   = std::variant<PrefillEvent, DecodeEvent, ReleaseEvent, MigrateEvent>;

struct WorkerStat{
  size_t index;

  int capacity;
  int inflight;
};

// Worker is responsible for managing a local runner on a device
// has a loop, pops commands, executes them, returns event results back to control plane
// commands : prefill, decode, release, migrate 
class Worker {
private:
  size_t index_ = 0;
  std::unique_ptr<DeviceAllocator> alloc_;
  std::unique_ptr<CachePool> pool_;
  std::unique_ptr<Runner> runner_;
  std::unique_ptr<CommAgent> comm_agent_;
  std::function<void(Event)> emit_;

  bool stop_ = false;

  std::mutex cmd_mu_;
  std::deque<Command> cmdq_;
  std::condition_variable cmd_cv_;
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
  PrefillEvent prefill_(uint64_t req_id, std::span<const int32_t> tokens);

  // one forward pass, update cache, and return new token
  DecodeEvent decode_(uint64_t req_id, const int32_t token);

  // Posts a transfer. On success stores the handle in pending_transfers_ and
  // returns nullopt (MigrateEvent is emitted later by poll_transfer_states_).
  // On immediate failure returns a MigrateEvent to emit now.
  std::optional<MigrateEvent> migrate_(const MigrateCmd& cmd);
  std::optional<MigrateEvent> migrate_recv_(const MigrateCmd& cmd);
  std::optional<MigrateEvent> migrate_send_(const MigrateCmd& cmd);
  std::optional<MigrateEvent> post_xfer_(const MigrateCmd& cmd, int64_t seq_len);

  // loops through all the pending transfers, when any transfer is successfull or failed
  // it simply queues the appropriate events into the worker events
  void poll_transfer_states_();

  // loop: wait on cv, pop a command, execute it, emit the resulting event.
  // Exits once stop_ is set and the queue has drained.
  void loop_();

public:
  // Spawns the worker's own processing thread; events are pushed to `emit`
  // as they're produced (called from the worker thread).
  Worker(size_t index, std::unique_ptr<Runner> runner, std::unique_ptr<DeviceAllocator> alloc,
         std::unique_ptr<CachePool> pool, std::unique_ptr<CommAgent> comm_agent,
         std::function<void(Event)> emit);

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Signals the loop to stop and joins the thread.
  ~Worker();

  void enqueue(Command cmd);
};
