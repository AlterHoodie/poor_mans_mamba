#pragma once

#include "comm/comm_agent.h"
#include "comm/nccl_comm_agent.h"
#include "comm/nixl_comm_agent.h"
#include "comm/transport_control_plane.h"
#include "runtime/cluster_config.h"

#include <memory>

// Optional in-process transport state for callers that drive several agents from
// one address space without a ClusterScheduler (benches, unit tests). It lets NIXL
// peers share a metadata directory directly. ClusterScheduler workers never use it:
// NCCL, MemcpyPeer and NIXL all coordinate through the transport control plane, so
// workers can live in threads or in separate processes.
struct TransportContext {
  TransportBackend backend = TransportBackend::MemcpyPeer;
  int n_workers = 0;
  std::shared_ptr<NixlCluster> nixl;
};

// Validate the ClusterConfig worker_mode x device x transport matrix.
Status validate_cluster_config(const ClusterConfig& cfg);

// Validate ClusterConfig device×transport matrix and build shared context.
StatusOr<std::shared_ptr<TransportContext>> create_transport_context(const ClusterConfig& cfg);

// Per-worker CommAgent. ctx may be null; Nixl then keeps a private metadata
// directory that the control plane populates (peer_md / peers_finalized).
StatusOr<std::unique_ptr<CommAgent>> create_comm_agent(const ClusterConfig& cfg, int device_id,
                                                       const std::shared_ptr<TransportContext>& ctx);

// Parent-side control-plane brain for cfg.transport. Fails if the backend cannot
// set up its bootstrap state (e.g. ncclGetUniqueId).
StatusOr<std::unique_ptr<TransportControlPlane>>
create_transport_control_plane(const ClusterConfig& cfg);

// After all Nixl agents are created and have register_slab'd, exchange peer metadata.
Status finalize_transport_peers(const std::shared_ptr<TransportContext>& ctx);
