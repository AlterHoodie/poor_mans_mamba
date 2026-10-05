#pragma once

#include "core/status.h"
#include "runtime/cluster_config.h"
#include "runtime/worker_slot.h"

#include <vector>

// Host-side worker creation. Validates `cfg`, then starts cfg.n_workers workers (device
// ids 0..n-1) as threads or child processes per cfg.worker_mode. Each worker begins
// loading its own model replica immediately, so loads overlap; readiness is awaited by
// ClusterScheduler::start, which takes ownership of the returned slots.
//
// Process mode needs the host binary to call maybe_run_worker_process() first in main().
StatusOr<std::vector<WorkerSlot>> create_cluster_workers(const ClusterConfig& cfg);
