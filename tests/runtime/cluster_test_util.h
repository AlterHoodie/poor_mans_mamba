#pragma once

#include "core/status.h"
#include "runtime/cluster_config.h"
#include "runtime/cluster_scheduler.h"
#include "worker_factory.h"

#include <utility>
#include <vector>

// Host bring-up for tests: create the workers, then hand them to the scheduler.
inline Status start_cluster(ClusterScheduler& sched, const ClusterConfig& cfg) {
  StatusOr<std::vector<WorkerSlot>> workers_or = create_cluster_workers(cfg);
  if (!workers_or.ok())
    return workers_or.status();
  return sched.start(cfg, std::move(workers_or.value()));
}
