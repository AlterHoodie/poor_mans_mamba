#pragma once

#include "comm/comm_agent.h"
#include "core/device.h"

#include <cstdint>
#include <unordered_map>

// Direct slot→slot copy between workers in one process.
// - Device::CPU: std::memcpy
// - Device::GPU: cudaMemcpyPeer (or D2D when src/dst share a device)
//
// Receiver-driven: Recv copies from XferDesc::remote_ptr (src VA from MigrateCmd)
// into local_ptr. Send posts return Done immediately (hold src until Release).
class MemcpyPeerCommAgent : public CommAgent {
public:
  MemcpyPeerCommAgent(int device_id, Device kind);
  ~MemcpyPeerCommAgent() override;

  Status register_slab(void* ptr, size_t bytes) override;
  StatusOr<PostResult> post(XferDesc& desc) override;
  XferState poll(const XferHandle& handle) override;
  int64_t xfer_seq_len(const XferHandle& handle) override;
  void shutdown() override;

private:
  struct Entry {
    XferRole role = XferRole::Send;
    XferState state = XferState::Pending;
    int64_t seq_len = 0;
  };

  Device kind_;
  std::unordered_map<uint64_t, Entry> xfers_;

  Status ensure_peer_access_(int peer_device_id);
  Status do_copy_(void* dst, int dst_dev, void* src, int src_dev, size_t bytes);
};
