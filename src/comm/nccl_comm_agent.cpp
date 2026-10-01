#include "comm/nccl_comm_agent.h"

#include <string>
#include <utility>

#if MAMBASERVE_WITH_NCCL
namespace {

Status nccl_to_status(ncclResult_t r, const char* what) {
  if (r == ncclSuccess)
    return Status::Ok();
  return Status::RuntimeError(std::string(what) + ": " + ncclGetErrorString(r));
}

} // namespace
#endif

NcclCommAgent::NcclCommAgent(int device_id, int rank, std::shared_ptr<NcclCluster> cluster)
    : CommAgent(device_id), rank_(rank), cluster_(std::move(cluster)) {
#if MAMBASERVE_WITH_NCCL
  if (!cluster_ || cluster_->nranks <= 0)
    return;
  if (rank_ < 0 || rank_ >= cluster_->nranks)
    return;

  if (cudaSetDevice(device_id()) != cudaSuccess)
    return;
  if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
    stream_ = nullptr;
    return;
  }
  if (ncclCommInitRank(&comm_, cluster_->nranks, cluster_->id, rank_) != ncclSuccess) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
    comm_ = nullptr;
  }
#else
  (void)rank_;
#endif
}

NcclCommAgent::~NcclCommAgent() { shutdown(); }

Status NcclCommAgent::register_slab(void*, size_t) { return Status::Ok(); }

StatusOr<XferHandle> NcclCommAgent::post(XferDesc& desc) {
#if !MAMBASERVE_WITH_NCCL
  (void)desc;
  return Status::NotImplemented("NCCL transport not enabled (build with MAMBASERVE_WITH_NCCL=ON)");
#else
  if (!comm_ || !stream_)
    return Status::RuntimeError("NcclCommAgent not initialized");
  if (desc.local_ptr == nullptr || desc.bytes == 0)
    return Status::InvalidArgument("NCCL xfer requires local_ptr and bytes");
  if (!cluster_ || desc.peer_device_id < 0 || desc.peer_device_id >= cluster_->nranks)
    return Status::InvalidArgument("invalid peer_device_id for NCCL");

  const int peer = desc.peer_device_id;
  if (peer == rank_)
    return Status::InvalidArgument("NCCL peer cannot be self");

  if (cudaSetDevice(device_id()) != cudaSuccess)
    return Status::RuntimeError("cudaSetDevice failed");

  int64_t* seq_dev = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&seq_dev), sizeof(int64_t)) != cudaSuccess)
    return Status::RuntimeError("cudaMalloc seq_len header failed");

  if (desc.role == XferRole::Send) {
    // transfer seq_len from host to device
    if (cudaMemcpyAsync(seq_dev, &desc.seq_len, sizeof(int64_t), cudaMemcpyHostToDevice, stream_) !=
        cudaSuccess) {
      cudaFree(seq_dev);
      return Status::RuntimeError("cudaMemcpyAsync seq_len failed");
    }
  }

  // treat the next batch of p2p transfers as one unit/group
  ncclResult_t nr = ncclGroupStart();
  if (nr != ncclSuccess) {
    cudaFree(seq_dev);
    return nccl_to_status(nr, "ncclGroupStart");
  }

  if (desc.role == XferRole::Send) {
    // send commands
    nr = ncclSend(seq_dev, 1, ncclInt64, peer, comm_, stream_);
    if (nr == ncclSuccess)
      nr = ncclSend(desc.local_ptr, static_cast<size_t>(desc.bytes), ncclUint8, peer, comm_,
                    stream_);
  } else {
    // recieve commands
    nr = ncclRecv(seq_dev, 1, ncclInt64, peer, comm_, stream_);
    if (nr == ncclSuccess)
      nr = ncclRecv(desc.local_ptr, static_cast<size_t>(desc.bytes), ncclUint8, peer, comm_,
                    stream_);
  }

  ncclResult_t end = ncclGroupEnd();
  if (nr != ncclSuccess || end != ncclSuccess) {
    cudaFree(seq_dev);
    if (nr != ncclSuccess)
      return nccl_to_status(nr, "ncclSend/Recv");
    return nccl_to_status(end, "ncclGroupEnd");
  }

  // Send: ack as soon as ops are submitted. Cluster still waits for Recv's
  // completion ack before releasing src. Recv: track stream completion.
  cudaEvent_t ev = nullptr;
  XferState initial = XferState::Done;
  if (desc.role == XferRole::Recv) {
    initial = XferState::Pending;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
      cudaFree(seq_dev);
      return Status::RuntimeError("cudaEventCreate failed");
    }
    if (cudaEventRecord(ev, stream_) != cudaSuccess) {
      cudaEventDestroy(ev);
      cudaFree(seq_dev);
      return Status::RuntimeError("cudaEventRecord failed");
    }
  }

  const uint64_t hid = id_counter_.fetch_add(1, std::memory_order_relaxed);
  Entry e;
  e.role = desc.role;
  e.state = initial;
  e.seq_len = desc.seq_len;
  e.done_event = ev;
  e.seq_dev = seq_dev;
  xfers_[hid] = e;
  return XferHandle{hid};
#endif
}

XferState NcclCommAgent::poll(const XferHandle& handle) {
#if !MAMBASERVE_WITH_NCCL
  (void)handle;
  return XferState::Error;
#else
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return XferState::Error;
  Entry& e = it->second;
  if (e.state != XferState::Pending)
    return e.state; // Send is Done after post; Recv Done/Error after event

  // Recv: wait until stream work (ncclRecv) finishes, then pull seq_len.
  cudaError_t q = cudaEventQuery(e.done_event);
  if (q == cudaErrorNotReady)
    return XferState::Pending;
  if (q != cudaSuccess) {
    e.state = XferState::Error;
    return e.state;
  }

  if (e.seq_dev != nullptr) {
    int64_t sl = 0;
    // copy the seqlen back to host once transfer is finished
    if (cudaMemcpy(&sl, e.seq_dev, sizeof(int64_t), cudaMemcpyDeviceToHost) == cudaSuccess)
      e.seq_len = sl;
    else {
      e.state = XferState::Error;
      return e.state;
    }
  }

  e.state = XferState::Done;
  return e.state;
#endif
}

int64_t NcclCommAgent::xfer_seq_len(const XferHandle& handle) {
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return 0;
  return it->second.seq_len;
}

void NcclCommAgent::shutdown() {
#if MAMBASERVE_WITH_NCCL
  for (auto& kv : xfers_) {
    Entry& e = kv.second;
    if (e.done_event) {
      cudaEventDestroy(e.done_event);
      e.done_event = nullptr;
    }
    if (e.seq_dev) {
      cudaFree(e.seq_dev);
      e.seq_dev = nullptr;
    }
  }
  xfers_.clear();
  if (comm_) {
    ncclCommDestroy(comm_);
    comm_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
#else
  xfers_.clear();
#endif
}
