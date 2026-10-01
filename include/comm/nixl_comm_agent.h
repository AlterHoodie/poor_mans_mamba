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

// Recv publishes the destination slot so Send can build remote WRITE descs.
struct SlotAnnounce {
  void* dst_ptr = nullptr;
  size_t bytes = 0;
  int dst_device = -1;
  bool ready = false;
};

// In-process NIXL peer directory: agent metadata keyed by device_id.
struct NixlCluster {
  int n_workers = 0;
  std::mutex mu;
  // device_id -> getLocalMD blob (empty until register_slab).
  std::unordered_map<int, std::string> local_md;
  // xfer_id -> Recv-published destination slot.
  std::unordered_map<uint64_t, SlotAnnounce> slots;
  bool peers_finalized = false;
};

// Register-once / transfer-many NIXL backend (UCX WRITE from Send).
// When MAMBASERVE_WITH_NIXL is off, register_slab/post return NotImplemented.
class NixlCommAgent : public CommAgent {
public:
  NixlCommAgent(int device_id, std::shared_ptr<NixlCluster> cluster);
  ~NixlCommAgent() override;

  Status register_slab(void* ptr, size_t bytes) override;
  StatusOr<XferHandle> post(XferDesc& desc) override;
  XferState poll(const XferHandle& handle) override;
  int64_t xfer_seq_len(const XferHandle& handle) override;
  void shutdown() override;

  static std::string agent_name(int device_id);

private:
  struct Entry {
    XferRole role = XferRole::Send;
    XferState state = XferState::Pending;
    int64_t seq_len = 0;
    uint64_t xfer_id = 0;
    void* local_ptr = nullptr;
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
  // Notifs drained from NIXL but not yet matched to a Recv Entry.
  std::vector<std::pair<std::string, nixl_blob_t>> pending_notifs_;

  Status init_backend_();
  Status ensure_peers_loaded_();
  XferState poll_send_(Entry& e);
  XferState poll_recv_(Entry& e);
  static nixl_blob_t pack_notif_(uint64_t xfer_id, int64_t seq_len);
  static bool unpack_notif_(const nixl_blob_t& blob, uint64_t* xfer_id, int64_t* seq_len);
#endif
};
