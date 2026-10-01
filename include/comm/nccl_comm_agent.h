#pragma once

#include "comm/comm_agent.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

#if MAMBASERVE_WITH_NCCL
#include <nccl.h>
#include <cuda_runtime.h>
#endif

// Shared NCCL bootstrap state created once per cluster load.
struct NcclCluster {
  int nranks = 0;
#if MAMBASERVE_WITH_NCCL
  ncclUniqueId id{};
#endif
};

// GPU point-to-point migrate via ncclSend / ncclRecv on a shared communicator.
// When MAMBASERVE_WITH_NCCL is off, construction succeeds but post returns NotImplemented.
class NcclCommAgent : public CommAgent {
public:
  NcclCommAgent(int device_id, int rank, std::shared_ptr<NcclCluster> cluster);
  ~NcclCommAgent() override;

  Status register_slab(void* ptr, size_t bytes) override;
  StatusOr<XferHandle> post(XferDesc& desc) override;
  XferState poll(const XferHandle& handle) override;
  int64_t xfer_seq_len(const XferHandle& handle) override;
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
  std::shared_ptr<NcclCluster> cluster_;
  std::unordered_map<uint64_t, Entry> xfers_;

#if MAMBASERVE_WITH_NCCL
  ncclComm_t comm_ = nullptr;
  cudaStream_t stream_ = nullptr;
#endif
};
