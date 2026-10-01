#include "comm/comm_factory.h"

#include "comm/memcpy_peer_comm_agent.h"

#include <string>
#include <utility>

#if !defined(MAMBASERVE_WITH_CUDA)
#define MAMBASERVE_WITH_CUDA 0
#endif
#if !defined(MAMBASERVE_WITH_NCCL)
#define MAMBASERVE_WITH_NCCL 0
#endif
#if !defined(MAMBASERVE_WITH_NIXL)
#define MAMBASERVE_WITH_NIXL 0
#endif

namespace {

Status validate_cluster_config(const ClusterConfig& cfg) {
  if (cfg.model_dir.empty())
    return Status::InvalidArgument("model_dir cannot be empty");
  if (cfg.n_workers <= 0)
    return Status::InvalidArgument("n_workers must be > 0");
  if (cfg.num_slots <= 0)
    return Status::InvalidArgument("num_slots must be > 0");

  if (cfg.device == Device::CPU) {
    if (cfg.transport != TransportBackend::MemcpyPeer)
      return Status::InvalidArgument("Device::CPU only supports TransportBackend::MemcpyPeer");
    return Status::Ok();
  }

  if (cfg.device == Device::GPU) {
#if !MAMBASERVE_WITH_CUDA
    return Status::InvalidArgument("Cannot create gpu workers in a non gpu host");
#endif
    if (cfg.transport == TransportBackend::Nccl) {
#if !MAMBASERVE_WITH_NCCL
      return Status::NotImplemented("NCCL transport not enabled (build with MAMBASERVE_WITH_NCCL=ON)");
#endif
      return Status::Ok();
    }
    if (cfg.transport == TransportBackend::Nixl) {
#if !MAMBASERVE_WITH_NIXL
      return Status::NotImplemented("NIXL transport not enabled (build with MAMBASERVE_WITH_NIXL=ON)");
#endif
      return Status::Ok();
    }
    if (cfg.transport == TransportBackend::MemcpyPeer)
      return Status::Ok();
    return Status::InvalidArgument("unknown transport backend");
  }

  return Status::InvalidArgument("unsupported device kind");
}

} // namespace

StatusOr<std::shared_ptr<TransportContext>> create_transport_context(const ClusterConfig& cfg) {
  if (Status s = validate_cluster_config(cfg); !s.ok())
    return s;

  auto ctx = std::make_shared<TransportContext>();
  ctx->backend = cfg.transport;
  ctx->n_workers = cfg.n_workers;

  if (cfg.transport == TransportBackend::Nccl) {
#if MAMBASERVE_WITH_NCCL
    auto nccl = std::make_shared<NcclCluster>();
    nccl->nranks = cfg.n_workers;
    ncclResult_t r = ncclGetUniqueId(&nccl->id);
    if (r != ncclSuccess)
      return Status::RuntimeError(std::string("ncclGetUniqueId: ") + ncclGetErrorString(r));
    ctx->nccl = std::move(nccl);
#else
    return Status::NotImplemented("NCCL transport not enabled");
#endif
  } else if (cfg.transport == TransportBackend::Nixl) {
#if MAMBASERVE_WITH_NIXL
    auto nixl = std::make_shared<NixlCluster>();
    nixl->n_workers = cfg.n_workers;
    ctx->nixl = std::move(nixl);
#else
    return Status::NotImplemented("NIXL transport not enabled");
#endif
  }

  return ctx;
}

StatusOr<std::unique_ptr<CommAgent>> create_comm_agent(const ClusterConfig& cfg, int device_id,
                                                       const std::shared_ptr<TransportContext>& ctx) {
  if (device_id < 0)
    return Status::InvalidArgument("device id cannot be less than 0");
  if (Status s = validate_cluster_config(cfg); !s.ok())
    return s;

  switch (cfg.transport) {
  case TransportBackend::MemcpyPeer:
    return std::unique_ptr<CommAgent>(std::make_unique<MemcpyPeerCommAgent>(device_id, cfg.device));

  case TransportBackend::Nccl: {
#if MAMBASERVE_WITH_NCCL
    if (!ctx || !ctx->nccl)
      return Status::InvalidArgument("Nccl TransportContext missing");
    auto agent = std::make_unique<NcclCommAgent>(device_id, /*rank=*/device_id, ctx->nccl);
    return std::unique_ptr<CommAgent>(std::move(agent));
#else
    (void)ctx;
    return Status::NotImplemented("NCCL transport not enabled");
#endif
  }

  case TransportBackend::Nixl: {
#if MAMBASERVE_WITH_NIXL
    if (!ctx || !ctx->nixl)
      return Status::InvalidArgument("Nixl TransportContext missing");
    // Metadata is published inside register_slab after registerMem.
    return std::unique_ptr<CommAgent>(std::make_unique<NixlCommAgent>(device_id, ctx->nixl));
#else
    (void)ctx;
    return Status::NotImplemented("NIXL transport not enabled");
#endif
  }
  }

  return Status::InvalidArgument("unknown transport backend");
}

Status finalize_transport_peers(const std::shared_ptr<TransportContext>& ctx) {
  if (!ctx)
    return Status::Ok();
  if (ctx->backend != TransportBackend::Nixl)
    return Status::Ok();
  if (!ctx->nixl)
    return Status::InvalidArgument("Nixl TransportContext missing");

  std::lock_guard<std::mutex> lk(ctx->nixl->mu);
  if (static_cast<int>(ctx->nixl->local_md.size()) != ctx->nixl->n_workers)
    return Status::RuntimeError("NIXL local metadata missing for some workers");
  for (const auto& kv : ctx->nixl->local_md) {
    if (kv.second.empty())
      return Status::RuntimeError("NIXL local metadata empty for a worker");
  }

  // Peers call loadRemoteMD lazily on first post from these blobs.
  ctx->nixl->peers_finalized = true;
  return Status::Ok();
}
