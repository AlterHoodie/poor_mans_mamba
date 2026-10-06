#pragma once

#include "comm/comm_agent.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#if MAMBASERVE_WITH_NIXL
#include "nixl.h"
#endif

// NIXL peer directory: agent metadata keyed by device_id. Each agent owns one (private
// per worker under ClusterScheduler, filled by peer_md / peers_finalized from the
// parent), or several agents in one process may share one (benches/tests, see
// TransportContext).
struct NixlCluster {
  int n_workers = 0;
  std::mutex mu;
  // device_id -> getLocalMD blob (empty until register_slab).
  std::unordered_map<int, std::string> local_md;
  bool peers_finalized = false;
};

// Register-once / transfer-many NIXL backend (UCX READ from Recv).
// Send posts return Done immediately (hold src until Release). Recv drives
// NIXL_READ using remote_ptr from MigrateCmd. When MAMBASERVE_WITH_NIXL is off,
// register_slab/post return NotImplemented.
class NixlCommAgent : public CommAgent {
public:
  NixlCommAgent(int device_id, std::shared_ptr<NixlCluster> cluster);
  ~NixlCommAgent() override;

  Status register_slab(void* ptr, size_t bytes) override;
  StatusOr<PostResult> post(XferDesc& desc) override;
  XferState poll(const XferHandle& handle) override;
  int64_t xfer_seq_len(const XferHandle& handle) override;
  // After register_slab: publish this agent's metadata up to the parent (publish_md).
  std::optional<mambaserve::TransportControl> make_register_announce() override;
  // peer_md / peers_finalized from the parent.
  Status handle_transport(const mambaserve::TransportControl& msg) override;
  void shutdown() override;

  static std::string agent_name(int device_id);

private:
  struct Entry {
    XferRole role = XferRole::Send;
    XferState state = XferState::Pending;
    int64_t seq_len = 0;
    uint64_t xfer_id = 0;
    void* local_ptr = nullptr;
    void* remote_ptr = nullptr;
    size_t bytes = 0;
    int peer_device_id = -1;
#if MAMBASERVE_WITH_NIXL
    nixlXferReqH* req = nullptr;
    bool posted = false;
#endif
  };

  std::shared_ptr<NixlCluster> cluster_;
  void* slab_ptr_ = nullptr;
  size_t slab_bytes_ = 0;
  bool peers_loaded_ = false;
  std::unordered_map<uint64_t, Entry> xfers_;

#if MAMBASERVE_WITH_NIXL
  std::unique_ptr<nixlAgent> agent_;
  nixlBackendH* backend_ = nullptr;
  nixlBlobDesc slab_desc_{};
  bool slab_registered_ = false;

  Status init_backend_();
  Status ensure_peers_loaded_();
  XferState poll_recv_(Entry& e);
  XferState create_xfer_req_(Entry& e);
#endif
};
