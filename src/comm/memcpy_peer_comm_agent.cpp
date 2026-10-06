#include "comm/memcpy_peer_comm_agent.h"

#include "core/status.h"

#include <cstring>
#include <string>

#ifdef MAMBASERVE_WITH_CUDA
#include <cuda_runtime.h>
#endif

MemcpyPeerCommAgent::MemcpyPeerCommAgent(int device_id, Device kind)
    : CommAgent(device_id), kind_(kind) {}

MemcpyPeerCommAgent::~MemcpyPeerCommAgent() { shutdown(); }

Status MemcpyPeerCommAgent::register_slab(void* /*ptr*/, size_t /*bytes*/) { return Status::Ok(); }

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
    if (Status s = ensure_peer_access_(src_dev); !s.ok())
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

StatusOr<PostResult> MemcpyPeerCommAgent::post(XferDesc& desc) {
  if (desc.local_ptr == nullptr)
    return Status::InvalidArgument("local_ptr cannot be null");
  if (desc.bytes == 0)
    return Status::InvalidArgument("bytes must be > 0");

  // Send: hold src until Release; no wire work.
  if (desc.role == XferRole::Send)
    return PostResult{.handle = {}, .state = XferState::Done};

  if (desc.remote_ptr == nullptr)
    return Status::InvalidArgument("Memcpy Recv requires remote_ptr (src slot)");
  if (desc.peer_device_id < 0)
    return Status::InvalidArgument("invalid peer_device_id for Memcpy Recv");

  Status copy =
      do_copy_(desc.local_ptr, device_id(), desc.remote_ptr, desc.peer_device_id, desc.bytes);
  if (!copy.ok())
    return copy;

  const uint64_t id = id_counter_.fetch_add(1, std::memory_order_relaxed);
  xfers_[id] = Entry{
      .role = XferRole::Recv,
      .state = XferState::Done,
      .seq_len = desc.seq_len,
  };
  return PostResult{.handle = {.id = id}, .state = XferState::Done};
}

XferState MemcpyPeerCommAgent::poll(const XferHandle& handle) {
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return XferState::Error;
  return it->second.state;
}

int64_t MemcpyPeerCommAgent::xfer_seq_len(const XferHandle& handle) {
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return 0;
  return it->second.seq_len;
}

void MemcpyPeerCommAgent::shutdown() { xfers_.clear(); }
