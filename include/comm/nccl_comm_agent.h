#pragma once

#include "comm/comm_agent.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

#if MAMBASERVE_WITH_NCCL
#include <cuda_runtime.h>
#include <nccl.h>
#endif

// GPU point-to-point migrate via ncclSend / ncclRecv on a shared communicator.
//
// The communicator is created lazily from a NcclBootstrap control message
// (handle_transport), pushed by NcclTransportControl at cluster load. Until then
// post() fails. ncclCommInitRank blocks until every rank has joined, so each
// rank must handle its bootstrap on its own thread.
//
// When MAMBASERVE_WITH_NCCL is off, construction succeeds but post returns NotImplemented.
class NcclCommAgent : public CommAgent {
public:
  NcclCommAgent(int device_id, int rank, int nranks);
  ~NcclCommAgent() override;

  Status register_slab(void* ptr, size_t bytes) override;
  StatusOr<PostResult> post(XferDesc& desc) override;
  XferState poll(const XferHandle& handle) override;
  int64_t xfer_seq_len(const XferHandle& handle) override;
  // Init the communicator from NcclBootstrap; the ack is left for take_transport_reply().
  Status handle_transport(const mambaserve::TransportControl& msg) override;
  std::optional<mambaserve::TransportControl> take_transport_reply() override;
  void shutdown() override;

private:
  struct Entry {
    XferRole role = XferRole::Send;
    XferState state = XferState::Pending;
    int64_t seq_len = 0;
#if MAMBASERVE_WITH_NCCL
    cudaEvent_t done_event = nullptr;
    int64_t* seq_dev = nullptr;
#endif
  };

  int rank_ = -1;
  int nranks_ = 0;
  std::unordered_map<uint64_t, Entry> xfers_;
  std::optional<mambaserve::TransportControl> reply_;

#if MAMBASERVE_WITH_NCCL
  ncclComm_t comm_ = nullptr;
  cudaStream_t stream_ = nullptr;

  Status init_comm_(const mambaserve::NcclBootstrap& boot);
#endif
};
