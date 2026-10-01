#pragma once

#include "comm/comm_agent.h"
#include "comm/nccl_comm_agent.h"
#include "comm/nixl_comm_agent.h"
#include "runtime/cluster_config.h"

#include <memory>

// Opaque-ish shared transport state for one ClusterScheduler::load_model call.
struct TransportContext {
  TransportBackend backend = TransportBackend::MemcpyPeer;
  int n_workers = 0;
  std::shared_ptr<NcclCluster> nccl;
  std::shared_ptr<NixlCluster> nixl;
};

// Validate ClusterConfig device×transport matrix and build shared context.
StatusOr<std::shared_ptr<TransportContext>> create_transport_context(const ClusterConfig& cfg);

// Per-worker CommAgent. MemcpyPeer ignores ctx; Nccl/Nixl require matching ctx.
StatusOr<std::unique_ptr<CommAgent>> create_comm_agent(const ClusterConfig& cfg, int device_id,
                                                       const std::shared_ptr<TransportContext>& ctx);

// After all Nixl agents are created and have register_slab'd, exchange peer metadata.
Status finalize_transport_peers(const std::shared_ptr<TransportContext>& ctx);
