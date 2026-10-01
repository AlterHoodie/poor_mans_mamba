#include "comm/memcpy_peer_comm_agent.h"

#include "core/status.h"

#include <cstring>
#include <string>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

std::mutex MemcpyPeerCommAgent::rendezvous_mu_;
std::unordered_map<uint64_t, MemcpyPeerCommAgent::Rendezvous> MemcpyPeerCommAgent::rendezvous_;

MemcpyPeerCommAgent::MemcpyPeerCommAgent(int device_id, Device kind)
    : CommAgent(device_id), kind_(kind) {}

MemcpyPeerCommAgent::~MemcpyPeerCommAgent() { shutdown(); }

void MemcpyPeerCommAgent::publish_recv_(uint64_t xfer_id, void* dst, size_t bytes, int dst_dev) {
  // acquires lock 
  std::lock_guard<std::mutex> lk(rendezvous_mu_);
  // creates a new randezvous entry so later on send can poll on this xfer id 
  // and initiate a copy
  Rendezvous& rz = rendezvous_[xfer_id];
  rz.dst_ptr = dst;
  rz.bytes = bytes;
  rz.dst_device = dst_dev;
  rz.phase = Phase::Ready;
}

MemcpyPeerCommAgent::ClaimCopy MemcpyPeerCommAgent::try_claim_copy_(uint64_t xfer_id,
                                                                   size_t fallback_bytes) {
  // acquire lock
  std::lock_guard<std::mutex> lk(rendezvous_mu_);
  Rendezvous& rz = rendezvous_[xfer_id];
  
  // check phase of recv side whether its ready with dst ptr and and bytes
  ClaimCopy out;
  switch (rz.phase) {
  // recv side could not set dst ptr and bytes for some reason
  case Phase::Failed:
    out.claim = Claim::Error;
    return out;
  // waiting for recv side to post dst ptr and bytes
  case Phase::WaitingRecv:
  // copying the bytes, never runs currently cause its sync - will be useful when copy is async
  case Phase::Copying:
    out.claim = Claim::Pending;
    return out;
  // copy done 
  case Phase::Done:
    out.claim = Claim::Done;
    out.seq_len = rz.seq_len;
    return out;
  // recv side has posted valid dst ptr and bytes, can begin copying into it
  case Phase::Ready:
    rz.phase = Phase::Copying;
    out.claim = Claim::DoCopy;
    out.dst = rz.dst_ptr;
    out.bytes = rz.bytes != 0 ? rz.bytes : fallback_bytes;
    out.dst_dev = rz.dst_device;
    return out;
  }
  out.claim = Claim::Error;
  return out;
}

void MemcpyPeerCommAgent::finish_copy_(uint64_t xfer_id, int64_t seq_len, bool ok) {
  std::lock_guard<std::mutex> lk(rendezvous_mu_);
  auto it = rendezvous_.find(xfer_id);
  if (it == rendezvous_.end())
    return;
  Rendezvous& rz = it->second;
  if (!ok) {
    rz.phase = Phase::Failed;
    return;
  }
  rz.seq_len = seq_len;
  rz.phase = Phase::Done;
}

XferState MemcpyPeerCommAgent::poll_recv_state_(uint64_t xfer_id, int64_t* seq_len_out) {
  std::lock_guard<std::mutex> lk(rendezvous_mu_);
  auto it = rendezvous_.find(xfer_id);
  if (it == rendezvous_.end())
    return XferState::Pending;

  Rendezvous& rz = it->second;
  switch (rz.phase) {
  case Phase::Failed:
    return XferState::Error;
  case Phase::Done:
    if (seq_len_out)
      *seq_len_out = rz.seq_len;
    rendezvous_.erase(it);
    return XferState::Done;
  case Phase::WaitingRecv:
  case Phase::Ready:
  case Phase::Copying:
    return XferState::Pending;
  }
  return XferState::Error;
}

Status MemcpyPeerCommAgent::register_slab(void* /*ptr*/, size_t /*bytes*/) {
  return Status::Ok();
}

Status MemcpyPeerCommAgent::ensure_peer_access_(int peer_device_id) {
#ifdef MAMBASERVE_WITH_CUDA
  if (kind_ != Device::GPU)
    return Status::Ok();
  if (peer_device_id < 0)
    return Status::InvalidArgument("peer_device_id cannot be < 0");
  if (peer_device_id == device_id())
    return Status::Ok();

  cudaError_t err = cudaSetDevice(device_id());
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
  }
  // Without P2P support (e.g. consumer cards, T4 pairs on some hosts) skip the enable:
  // cudaMemcpyPeer still works and stages through host memory.
  int can_access = 0;
  if (cudaDeviceCanAccessPeer(&can_access, device_id(), peer_device_id) != cudaSuccess ||
      !can_access) {
    (void)cudaGetLastError();
    return Status::Ok();
  }
  err = cudaDeviceEnablePeerAccess(peer_device_id, 0);
  if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
    return Status::RuntimeError(std::string("cudaDeviceEnablePeerAccess failed: ") +
                                cudaGetErrorString(err));
  }
  (void)cudaGetLastError(); // clear sticky already-enabled
  return Status::Ok();
#else
  (void)peer_device_id;
  return Status::Ok();
#endif
}

Status MemcpyPeerCommAgent::do_copy_(void* dst, int dst_dev, void* src, int src_dev, size_t bytes) {
  if (dst == nullptr || src == nullptr)
    return Status::InvalidArgument("copy pointers cannot be null");
  if (bytes == 0)
    return Status::InvalidArgument("copy bytes must be > 0");
  // cpu mode use simple memcpy 
  if (kind_ == Device::CPU) {
    std::memcpy(dst, src, bytes);
    return Status::Ok();
  }

#ifdef MAMBASERVE_WITH_CUDA
  if (kind_ != Device::GPU)
    return Status::InvalidArgument("MemcpyPeerCommAgent: unsupported device kind");

  cudaError_t err = cudaSetDevice(device_id());
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
  }

  if (dst_dev == src_dev) {
    err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
  } else {
    if (Status s = ensure_peer_access_(dst_dev); !s.ok())
      return s;
    err = cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes);
  }
  if (err != cudaSuccess) {
    return Status::RuntimeError(std::string("GPU slot copy failed: ") + cudaGetErrorString(err));
  }
  return Status::Ok();
#else
  (void)dst_dev;
  (void)src_dev;
  return Status::InvalidArgument("GPU MemcpyPeer requires MAMBASERVE_WITH_CUDA");
#endif
}

StatusOr<XferHandle> MemcpyPeerCommAgent::post(XferDesc& desc) {
  if (desc.local_ptr == nullptr)
    return Status::InvalidArgument("local_ptr cannot be null");
  if (desc.bytes == 0)
    return Status::InvalidArgument("bytes must be > 0");

  if (desc.role == XferRole::Recv)
    // recieve side first creates the randevous xfer
    publish_recv_(desc.xfer_id, desc.local_ptr, desc.bytes, device_id());

  const uint64_t id = id_counter_.fetch_add(1, std::memory_order_relaxed);
  // local worker entries, to poll on status of requests 
  xfers_[id] = Entry{
      .desc = desc,
      .role = desc.role,
      .state = XferState::Pending,
      .xfer_id = desc.xfer_id,
      .seq_len = desc.seq_len,
  };
  return XferHandle{.id = id};
}

XferState MemcpyPeerCommAgent::poll_send_(Entry& entry) {
  // send side polling of xfer transfer request
  if (entry.state != XferState::Pending)
    return entry.state;

  ClaimCopy claim = try_claim_copy_(entry.xfer_id, entry.desc.bytes);
  switch (claim.claim) {
  case Claim::Pending:
    return XferState::Pending;
  case Claim::Error:
    entry.state = XferState::Error;
    return entry.state;
  case Claim::Done:
    entry.seq_len = claim.seq_len;
    entry.state = XferState::Done;
    return entry.state;
  case Claim::DoCopy:
    break;
  }

  Status copy = do_copy_(claim.dst, claim.dst_dev, entry.desc.local_ptr, device_id(), claim.bytes);
  finish_copy_(entry.xfer_id, entry.desc.seq_len, copy.ok());
  if (!copy.ok()) {
    entry.state = XferState::Error;
    return entry.state;
  }
  entry.seq_len = entry.desc.seq_len;
  entry.state = XferState::Done;
  return entry.state;
}

XferState MemcpyPeerCommAgent::poll_recv_(Entry& entry) {
  if (entry.state != XferState::Pending)
    return entry.state;

  int64_t seq_len = 0;
  const XferState st = poll_recv_state_(entry.xfer_id, &seq_len);
  if (st == XferState::Done) {
    entry.seq_len = seq_len;
    entry.state = XferState::Done;
  } else if (st == XferState::Error) {
    entry.state = XferState::Error;
  }
  return st;
}

// polls a particular xfer transfer request
XferState MemcpyPeerCommAgent::poll(const XferHandle& handle) {
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return XferState::Error;

  Entry& entry = it->second;
  return (entry.role == XferRole::Send) ? poll_send_(entry) : poll_recv_(entry);
}

int64_t MemcpyPeerCommAgent::xfer_seq_len(const XferHandle& handle) {
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return 0;
  return it->second.seq_len;
}

void MemcpyPeerCommAgent::shutdown() { xfers_.clear(); }
