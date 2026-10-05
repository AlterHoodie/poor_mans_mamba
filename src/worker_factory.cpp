#include "worker_factory.h"

#include "comm/comm_factory.h"
#include "runtime/ipc/process_comm.h"
#include "runtime/ipc/thread_comm.h"
#include "runtime/worker.h"
#include "worker_process.h"

#include <unistd.h>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

StatusOr<WorkerSlot> create_worker(const ClusterConfig& cfg, int device_id) {
  WorkerSlot slot;
  slot.device_id = device_id;

  if (cfg.worker_mode == WorkerMode::Process) {
    ProcessChannelFds fds;
    ASSIGN_OR_RETURN(fds, make_process_socketpair());

    StatusOr<std::unique_ptr<WorkerProcess>> proc_or =
        spawn_worker_process(cfg, static_cast<size_t>(device_id), device_id, fds.child_fd);
    // The child holds its own copy of its end now (or never started).
    ::close(fds.child_fd);
    if (!proc_or.ok()) {
      ::close(fds.parent_fd);
      return Status(proc_or.status());
    }
    slot.chan = std::make_unique<ProcessChannel>(fds.parent_fd);
    slot.process = std::move(proc_or.value());
    return slot;
  }

  auto [parent, child] = make_thread_channel_pair();
  slot.chan = std::move(parent);
  slot.worker = std::make_unique<Worker>(WorkerBootstrap{
      .index = static_cast<size_t>(device_id),
      .device_id = device_id,
      .cfg = cfg,
      .ipc = std::move(child),
  });
  return slot;
}

} // namespace

StatusOr<std::vector<WorkerSlot>> create_cluster_workers(const ClusterConfig& cfg) {
  if (Status s = validate_cluster_config(cfg); !s.ok())
    return s;

  std::vector<WorkerSlot> slots;
  slots.reserve(static_cast<size_t>(cfg.n_workers));
  for (int device_id = 0; device_id < cfg.n_workers; ++device_id) {
    StatusOr<WorkerSlot> slot_or = create_worker(cfg, device_id);
    if (!slot_or.ok())
      return Status(slot_or.status()); // slots dtor joins/reaps the ones already started
    slots.push_back(std::move(slot_or.value()));
  }
  return slots;
}
