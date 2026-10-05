#include "comm/nccl_comm_agent.h"

#include <cstring>
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

// Communicator init is deferred to handle_transport(NcclBootstrap).
NcclCommAgent::NcclCommAgent(int device_id, int rank, int nranks)
    : CommAgent(device_id), rank_(rank), nranks_(nranks) {}

NcclCommAgent::~NcclCommAgent() { shutdown(); }

Status NcclCommAgent::register_slab(void*, size_t) { return Status::Ok(); }

#if MAMBASERVE_WITH_NCCL
Status NcclCommAgent::init_comm_(const mambaserve::NcclBootstrap& boot) {
  if (comm_)
    return Status::InvalidArgument("NCCL communicator already initialized");
  if (boot.nranks() <= 0 || boot.rank() < 0 || boot.rank() >= boot.nranks())
    return Status::InvalidArgument("invalid NCCL bootstrap rank/nranks");
  if (boot.rank() != rank_ || boot.nranks() != nranks_)
    return Status::InvalidArgument("NCCL bootstrap does not match this agent's rank/nranks");

  ncclUniqueId id;
  if (boot.unique_id().size() != sizeof(id.internal))
    return Status::InvalidArgument("NCCL bootstrap unique_id has wrong size");
  std::memcpy(id.internal, boot.unique_id().data(), sizeof(id.internal));

  if (cudaSetDevice(device_id()) != cudaSuccess)
    return Status::RuntimeError("cudaSetDevice failed");
  if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
    stream_ = nullptr;
    return Status::RuntimeError("cudaStreamCreate failed");
  }
  // Blocks until every rank has joined.
  ncclResult_t r = ncclCommInitRank(&comm_, boot.nranks(), id, boot.rank());
  if (r != ncclSuccess) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
    comm_ = nullptr;
    return nccl_to_status(r, "ncclCommInitRank");
  }
  return Status::Ok();
}
#endif

Status NcclCommAgent::handle_transport(const mambaserve::TransportControl& msg) {
  if (msg.body_case() != mambaserve::TransportControl::kNcclCtrl)
    return Status::Ok();
  const mambaserve::NcclControl& ctrl = msg.nccl_ctrl();
  if (ctrl.body_case() != mambaserve::NcclControl::kBootstrap)
    return Status::Ok();

#if !MAMBASERVE_WITH_NCCL
  Status st = Status::NotImplemented("NCCL transport not enabled (build with MAMBASERVE_WITH_NCCL=ON)");
#else
  Status st = init_comm_(ctrl.bootstrap());
#endif

  mambaserve::TransportControl reply;
  auto* ack = reply.mutable_nccl_ctrl()->mutable_bootstrap_ack();
  ack->set_rank(rank_);
  ack->set_ok(st.ok());
  if (!st.ok())
    ack->set_error(st.message());
  reply_ = std::move(reply);
  return st;
}

std::optional<mambaserve::TransportControl> NcclCommAgent::take_transport_reply() {
  std::optional<mambaserve::TransportControl> out = std::move(reply_);
  reply_.reset();
  return out;
}

StatusOr<XferHandle> NcclCommAgent::post(XferDesc& desc) {
#if !MAMBASERVE_WITH_NCCL
  (void)desc;
  return Status::NotImplemented("NCCL transport not enabled (build with MAMBASERVE_WITH_NCCL=ON)");
#else
  if (!comm_ || !stream_)
    return Status::RuntimeError("NcclCommAgent not bootstrapped (no NcclBootstrap received)");
  if (desc.local_ptr == nullptr || desc.bytes == 0)
    return Status::InvalidArgument("NCCL xfer requires local_ptr and bytes");
  if (desc.peer_device_id < 0 || desc.peer_device_id >= nranks_)
    return Status::InvalidArgument("invalid peer_device_id for NCCL");

  const int peer = desc.peer_device_id;
  if (peer == rank_)
    return Status::InvalidArgument("NCCL peer cannot be self");

  if (cudaSetDevice(this->device_id()) != cudaSuccess)
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
