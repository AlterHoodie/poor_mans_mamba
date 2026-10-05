#pragma once

#include "comm/comm_agent.h"
#include "core/device.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>

// Direct slot→slot copy between workers in one process.
// - Device::CPU: std::memcpy
// - Device::GPU: cudaMemcpyPeer (or D2D when src/dst share a device)
//
// Send and Recv coordinate over the transport control plane, keyed by
// XferDesc::xfer_id:
//   Recv post  -> make_post_announce (dst slot)         -> Send.handle_transport
//   Send done  -> make_completion_announce (seq_len,ok) -> Recv.handle_transport
// so Recv lands the slot before Send copies, and seq_len is applied after the
// copy completes.
class MemcpyPeerCommAgent : public CommAgent {
public:
  MemcpyPeerCommAgent(int device_id, Device kind);
  ~MemcpyPeerCommAgent() override;

  Status register_slab(void* ptr, size_t bytes) override;
  StatusOr<XferHandle> post(XferDesc& desc) override;
  XferState poll(const XferHandle& handle) override;
  int64_t xfer_seq_len(const XferHandle& handle) override;
  // Recv: announce the destination slot to the Send peer via the parent.
  std::optional<mambaserve::TransportControl> make_post_announce(const XferDesc& desc) override;
  // Send: publish "copy finished" (seq_len, ok) to the Recv peer via the parent.
  std::optional<mambaserve::TransportControl>
  make_completion_announce(const XferHandle& handle, XferState state) override;
  // Send: record/clear announces; Recv: record completions routed from the parent.
  Status handle_transport(const mambaserve::TransportControl& msg) override;
  void shutdown() override;

private:
  struct Entry {
    XferDesc desc;
    XferRole role = XferRole::Send;
    XferState state = XferState::Pending;
    uint64_t xfer_id = 0;
    int64_t seq_len = 0;
    bool completion_sent = false;
  };

  // Recv-published destination slot (held by the Send agent).
  struct Announce {
    void* dst_ptr = nullptr;
    size_t bytes = 0;
    int dst_device = -1;
  };

  // Send-published copy result (held by the Recv agent).
  struct Completion {
    int64_t seq_len = 0;
    bool ok = false;
  };

  Device kind_;
  std::unordered_map<uint64_t, Entry> xfers_;

  // Written by handle_transport (worker thread, or a peer thread in benchmarks),
  // read by poll_send_/poll_recv_.
  std::mutex ctrl_mu_;
  std::unordered_map<uint64_t, Announce> announces_;
  std::unordered_map<uint64_t, Completion> completions_;

  Status ensure_peer_access_(int peer_device_id);
  Status do_copy_(void* dst, int dst_dev, void* src, int src_dev, size_t bytes);
  XferState poll_send_(Entry& entry);
  XferState poll_recv_(Entry& entry);
};
