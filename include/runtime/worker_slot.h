#pragma once

#include "runtime/ipc/ipc_comm.h"
#include "runtime/worker.h"
#include "worker_process.h"

#include <memory>

// Parent-side handle to one worker. Exactly one of `worker` (thread mode) and `process`
// (process mode) is set. Members are destroyed bottom-up: `chan` closes first so the
// worker sees EOF, then `process` reaps the child / `worker` joins its thread.
//
// Slots are built by the host (see create_cluster_workers in worker_factory.h) and
// handed to ClusterScheduler::start, which only schedules over them.
struct WorkerSlot {
  int device_id = -1;
  std::unique_ptr<Worker> worker;         // thread mode: in-process Worker owning a thread
  std::unique_ptr<WorkerProcess> process; // process mode: spawned child process
  std::unique_ptr<IpcChannel> chan;       // parent end
};
