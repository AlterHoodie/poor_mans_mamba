#pragma once

#include "comm/comm_agent.h"
#include "core/device.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>

// Direct slot→slot copy between workers in one process.
// - Device::CPU: std::memcpy
// - Device::GPU: cudaMemcpyPeer (or D2D when src/dst share a device)
//
// Send and Recv rendezvous on XferDesc::xfer_id so Recv can land the slot
// before Send copies, and seq_len can be applied after the copy completes.
class MemcpyPeerCommAgent : public CommAgent {
public:
  MemcpyPeerCommAgent(int device_id, Device kind);
  ~MemcpyPeerCommAgent() override;

  Status register_slab(void* ptr, size_t bytes) override;
  StatusOr<XferHandle> post(XferDesc& desc) override;
  XferState poll(const XferHandle& handle) override;
  int64_t xfer_seq_len(const XferHandle& handle) override;
  void shutdown() override;

private:
  enum class Phase { WaitingRecv, Ready, Copying, Done, Failed };

  struct Rendezvous {
    Phase phase = Phase::WaitingRecv;
    void* dst_ptr = nullptr;
    size_t bytes = 0;
    int dst_device = -1;
    int64_t seq_len = 0;
  };

  struct Entry {
    XferDesc desc;
    XferRole role = XferRole::Send;
    XferState state = XferState::Pending;
    uint64_t xfer_id = 0;
    int64_t seq_len = 0;
  };

  // Result of try_claim_copy_: either run the copy, or a terminal/wait state.
  enum class Claim { DoCopy, Pending, Done, Error };

  struct ClaimCopy {
    Claim claim = Claim::Pending;
    void* dst = nullptr;
    size_t bytes = 0;
    int dst_dev = -1;
    int64_t seq_len = 0; // valid when claim == Done
  };

  Device kind_;
  std::unordered_map<uint64_t, Entry> xfers_;

  static std::mutex rendezvous_mu_;
  static std::unordered_map<uint64_t, Rendezvous> rendezvous_;

  static void publish_recv_(uint64_t xfer_id, void* dst, size_t bytes, int dst_dev);
  static ClaimCopy try_claim_copy_(uint64_t xfer_id, size_t fallback_bytes);
  static void finish_copy_(uint64_t xfer_id, int64_t seq_len, bool ok);
  // On Done, erases the rendezvous entry under the same lock.
  static XferState poll_recv_state_(uint64_t xfer_id, int64_t* seq_len_out);

  Status ensure_peer_access_(int peer_device_id);
  Status do_copy_(void* dst, int dst_dev, void* src, int src_dev, size_t bytes);
  XferState poll_send_(Entry& entry);
  XferState poll_recv_(Entry& entry);
};
