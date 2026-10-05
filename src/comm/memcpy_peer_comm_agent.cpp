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

  const uint64_t id = id_counter_.fetch_add(1, std::memory_order_relaxed);
  // local worker entries, to poll on status of requests
  // (Recv's slot reaches Send via make_post_announce; Send's completion reaches
  // Recv via make_completion_announce - both routed by the control plane)
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

  // wait for the Recv side to announce its destination slot
  Announce ann;
  {
    std::lock_guard<std::mutex> lk(ctrl_mu_);
    auto it = announces_.find(entry.xfer_id);
    if (it == announces_.end())
      return XferState::Pending;
    ann = it->second;
    announces_.erase(it);
  }

  const size_t bytes = ann.bytes != 0 ? ann.bytes : entry.desc.bytes;
  Status copy = do_copy_(ann.dst_ptr, ann.dst_device, entry.desc.local_ptr, device_id(), bytes);
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

  // wait for the Send side to report that the copy finished
  Completion done;
  {
    std::lock_guard<std::mutex> lk(ctrl_mu_);
    auto it = completions_.find(entry.xfer_id);
    if (it == completions_.end())
      return XferState::Pending;
    done = it->second;
    completions_.erase(it);
  }

  if (!done.ok) {
    entry.state = XferState::Error;
    return entry.state;
  }
  entry.seq_len = done.seq_len;
  entry.state = XferState::Done;
  return entry.state;
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

std::optional<mambaserve::TransportControl>
MemcpyPeerCommAgent::make_post_announce(const XferDesc& desc) {
  if (desc.role != XferRole::Recv)
    return std::nullopt;
  mambaserve::TransportControl msg;
  auto* a = msg.mutable_memcpy_ctrl()->mutable_announce();
  a->set_xfer_id(desc.xfer_id);
  a->set_dst_ptr(reinterpret_cast<uint64_t>(desc.local_ptr));
  a->set_bytes(desc.bytes);
  a->set_dst_device(device_id());
  return msg;
}

std::optional<mambaserve::TransportControl>
MemcpyPeerCommAgent::make_completion_announce(const XferHandle& handle, XferState state) {
  if (state == XferState::Pending)
    return std::nullopt;
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return std::nullopt;
  Entry& entry = it->second;
  if (entry.role != XferRole::Send || entry.completion_sent)
    return std::nullopt;
  entry.completion_sent = true;

  mambaserve::TransportControl msg;
  auto* d = msg.mutable_memcpy_ctrl()->mutable_done();
  d->set_xfer_id(entry.xfer_id);
  d->set_seq_len(entry.seq_len);
  d->set_ok(state == XferState::Done);
  return msg;
}

Status MemcpyPeerCommAgent::handle_transport(const mambaserve::TransportControl& msg) {
  if (msg.body_case() != mambaserve::TransportControl::kMemcpyCtrl)
    return Status::Ok();
  const mambaserve::MemcpyControl& ctrl = msg.memcpy_ctrl();

  std::lock_guard<std::mutex> lk(ctrl_mu_);
  switch (ctrl.body_case()) {
  case mambaserve::MemcpyControl::kAnnounce: {
    const auto& a = ctrl.announce();
    announces_[a.xfer_id()] = Announce{
        .dst_ptr = reinterpret_cast<void*>(a.dst_ptr()),
        .bytes = static_cast<size_t>(a.bytes()),
        .dst_device = a.dst_device(),
    };
    break;
  }
  case mambaserve::MemcpyControl::kClear:
    announces_.erase(ctrl.clear().xfer_id());
    break;
  case mambaserve::MemcpyControl::kDone: {
    const auto& d = ctrl.done();
    completions_[d.xfer_id()] = Completion{.seq_len = d.seq_len(), .ok = d.ok()};
    break;
  }
  case mambaserve::MemcpyControl::BODY_NOT_SET:
    break;
  }
  return Status::Ok();
}

void MemcpyPeerCommAgent::shutdown() {
  xfers_.clear();
  std::lock_guard<std::mutex> lk(ctrl_mu_);
  announces_.clear();
  completions_.clear();
}
