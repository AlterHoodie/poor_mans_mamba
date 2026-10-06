#include "comm/nixl_comm_agent.h"

#include "proto/worker.pb.h"

#include <cstring>
#include <string>
#include <utility>

#if MAMBASERVE_WITH_NIXL
#include <cuda_runtime.h>
#endif

std::string NixlCommAgent::agent_name(int device_id) {
  return "worker-" + std::to_string(device_id);
}

NixlCommAgent::NixlCommAgent(int device_id, std::shared_ptr<NixlCluster> cluster)
    : CommAgent(device_id), cluster_(std::move(cluster)) {
#if MAMBASERVE_WITH_NIXL
  (void)init_backend_();
#endif
}

NixlCommAgent::~NixlCommAgent() { shutdown(); }

#if MAMBASERVE_WITH_NIXL

namespace {

Status nixl_to_status(nixl_status_t st, const char* what) {
  if (st == NIXL_SUCCESS || st == NIXL_IN_PROG)
    return Status::Ok();
  return Status::RuntimeError(std::string(what) + ": " + nixlEnumStrings::statusStr(st));
}

} // namespace

Status NixlCommAgent::init_backend_() {
  nixlAgentConfig cfg(true);
  // create worker specific nixl agent
  agent_ = std::make_unique<nixlAgent>(agent_name(device_id()), cfg);

  nixl_b_params_t init;
  nixl_mem_list_t mems;
  nixl_status_t st = agent_->getPluginParams("UCX", mems, init);
  if (st != NIXL_SUCCESS)
    return nixl_to_status(st, "getPluginParams(UCX)");

  // assign backend to agent
  st = agent_->createBackend("UCX", init, backend_);
  if (st != NIXL_SUCCESS) {
    backend_ = nullptr;
    return nixl_to_status(st, "createBackend(UCX)");
  }
  return Status::Ok();
}

Status NixlCommAgent::ensure_peers_loaded_() {
  if (peers_loaded_)
    return Status::Ok();
  if (!agent_ || !cluster_)
    return Status::RuntimeError("NIXL agent not initialized");

  std::lock_guard<std::mutex> lk(cluster_->mu);
  if (!cluster_->peers_finalized)
    return Status::RuntimeError("NIXL peers not finalized");
  for (const auto& kv : cluster_->local_md) {
    if (kv.first == device_id())
      continue;
    std::string remote_name;
    nixl_status_t st = agent_->loadRemoteMD(kv.second, remote_name);
    if (st != NIXL_SUCCESS)
      return nixl_to_status(st, "loadRemoteMD");
  }
  peers_loaded_ = true;
  return Status::Ok();
}

#endif // MAMBASERVE_WITH_NIXL

Status NixlCommAgent::register_slab(void* ptr, size_t bytes) {
#if !MAMBASERVE_WITH_NIXL
  (void)ptr;
  (void)bytes;
  return Status::NotImplemented("NIXL transport not enabled (build with MAMBASERVE_WITH_NIXL=ON)");
#else
  if (!agent_ || !backend_)
    return Status::RuntimeError("NIXL agent/backend not initialized");
  if (ptr == nullptr || bytes == 0)
    return Status::InvalidArgument("NIXL register_slab requires ptr and bytes");
  if (!cluster_)
    return Status::InvalidArgument("NixlCluster missing");

  if (cudaSetDevice(device_id()) != cudaSuccess)
    return Status::RuntimeError("cudaSetDevice failed");

  // desc to register one entier gpu slab
  slab_desc_ = nixlBlobDesc(reinterpret_cast<uintptr_t>(ptr), bytes,
                            static_cast<uint64_t>(device_id()), nixl_blob_t{});
  // create a list container for vram descriptors
  nixl_reg_dlist_t dlist(VRAM_SEG);
  // add the blob to the list
  dlist.addDesc(slab_desc_);

  nixl_opt_args_t extra;
  // register with UCX backend
  extra.backends.push_back(backend_);
  // memory can be used for transfer - under the hood means pinning for RDMA
  nixl_status_t st = agent_->registerMem(dlist, &extra);
  if (st != NIXL_SUCCESS)
    return nixl_to_status(st, "registerMem");

  slab_ptr_ = ptr;
  slab_bytes_ = bytes;
  slab_registered_ = true;

  nixl_blob_t md;
  st = agent_->getLocalMD(md);
  if (st != NIXL_SUCCESS)
    return nixl_to_status(st, "getLocalMD");

  {
    std::lock_guard<std::mutex> lk(cluster_->mu);
    cluster_->local_md[device_id()] = std::move(md);
  }
  return Status::Ok();
#endif
}

#if MAMBASERVE_WITH_NIXL
XferState NixlCommAgent::create_xfer_req_(Entry& e) {
  if (Status s = ensure_peers_loaded_(); !s.ok()) {
    e.state = XferState::Error;
    return e.state;
  }
  if (e.remote_ptr == nullptr) {
    e.state = XferState::Error;
    return e.state;
  }
  if (cudaSetDevice(device_id()) != cudaSuccess) {
    e.state = XferState::Error;
    return e.state;
  }

  // local = destination (this Recv worker); remote = source on Send peer.
  nixl_xfer_dlist_t local(VRAM_SEG);
  local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(e.local_ptr), e.bytes,
                              static_cast<uint64_t>(device_id())));

  nixl_xfer_dlist_t remote(VRAM_SEG);
  remote.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(e.remote_ptr), e.bytes,
                               static_cast<uint64_t>(e.peer_device_id)));

  nixl_opt_args_t extra;
  extra.backends.push_back(backend_);

  const std::string remote_agent = agent_name(e.peer_device_id);
  nixl_status_t st = agent_->createXferReq(NIXL_READ, local, remote, remote_agent, e.req, &extra);
  if (st != NIXL_SUCCESS) {
    e.state = XferState::Error;
    return e.state;
  }

  st = agent_->postXferReq(e.req);
  if (st < 0) {
    agent_->releaseXferReq(e.req);
    e.req = nullptr;
    e.state = XferState::Error;
    return e.state;
  }
  e.posted = true;

  if (st == NIXL_SUCCESS) {
    agent_->releaseXferReq(e.req);
    e.req = nullptr;
    e.state = XferState::Done;
    return e.state;
  }
  // NIXL_IN_PROG: flush/completion is driven by getXferStatus().
  e.state = XferState::Pending;
  return e.state;
}
#endif

StatusOr<PostResult> NixlCommAgent::post(XferDesc& desc) {
#if !MAMBASERVE_WITH_NIXL
  (void)desc;
  return Status::NotImplemented("NIXL transport not enabled (build with MAMBASERVE_WITH_NIXL=ON)");
#else
  if (!agent_ || !backend_)
    return Status::RuntimeError("NIXL agent/backend not initialized");
  if (slab_ptr_ == nullptr || slab_bytes_ == 0)
    return Status::RuntimeError("NIXL slab not registered");
  if (desc.local_ptr == nullptr || desc.bytes == 0)
    return Status::InvalidArgument("NIXL xfer requires local_ptr and bytes");
  if (!cluster_)
    return Status::RuntimeError("NixlCluster missing");

  const auto local = static_cast<const char*>(desc.local_ptr);
  const auto base = static_cast<const char*>(slab_ptr_);
  if (local < base || local + desc.bytes > base + slab_bytes_)
    return Status::InvalidArgument("NIXL local_ptr outside registered slab");
  if (desc.peer_device_id < 0 || desc.peer_device_id == device_id())
    return Status::InvalidArgument("invalid peer_device_id for NIXL");

  // Nothing to post send-side for nixl
  if (desc.role == XferRole::Send)
    return PostResult{.handle = {}, .state = XferState::Done};

  if (desc.remote_ptr == nullptr)
    return Status::InvalidArgument("NIXL Recv requires remote_ptr (src slot)");

  const uint64_t hid = id_counter_.fetch_add(1, std::memory_order_relaxed);
  Entry& e = xfers_[hid];
  e.role = desc.role;
  e.state = XferState::Pending;
  e.seq_len = desc.seq_len;
  e.xfer_id = desc.xfer_id;
  e.local_ptr = desc.local_ptr;
  e.remote_ptr = desc.remote_ptr;
  e.bytes = desc.bytes;
  e.peer_device_id = desc.peer_device_id;
  e.req = nullptr;
  e.posted = false;

  e.state = create_xfer_req_(e);
  return PostResult{.handle = {.id = hid}, .state = e.state};
#endif
}

#if MAMBASERVE_WITH_NIXL

XferState NixlCommAgent::poll_recv_(Entry& e) {
  if (e.state != XferState::Pending)
    return e.state;

  if (!e.posted || e.req == nullptr) {
    e.state = XferState::Error;
    return e.state;
  }

  nixl_status_t st = agent_->getXferStatus(e.req);
  if (st == NIXL_IN_PROG)
    return XferState::Pending;
  agent_->releaseXferReq(e.req);
  e.req = nullptr;
  e.state = (st == NIXL_SUCCESS) ? XferState::Done : XferState::Error;
  return e.state;
}

#endif // MAMBASERVE_WITH_NIXL

XferState NixlCommAgent::poll(const XferHandle& handle) {
#if !MAMBASERVE_WITH_NIXL
  (void)handle;
  return XferState::Error;
#else
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return XferState::Error;
  return poll_recv_(it->second);
#endif
}

int64_t NixlCommAgent::xfer_seq_len(const XferHandle& handle) {
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return 0;
  return it->second.seq_len;
}

Status NixlCommAgent::handle_transport(const mambaserve::TransportControl& msg) {
  if (msg.body_case() != mambaserve::TransportControl::kNixlCtrl)
    return Status::Ok();
  const mambaserve::NixlControl& ctrl = msg.nixl_ctrl();

  switch (ctrl.body_case()) {
  // Parent fans out every other worker's agent metadata, then signals that the
  // directory is complete; peers are loaded lazily on the first post().
  case mambaserve::NixlControl::kPeerMd: {
    if (!cluster_)
      return Status::RuntimeError("NixlCluster missing");
    const mambaserve::NixlLocalMd& peer = ctrl.peer_md();
    if (peer.device_id() == device_id())
      return Status::Ok();
    std::lock_guard<std::mutex> lk(cluster_->mu);
    cluster_->local_md[peer.device_id()] = peer.md();
    return Status::Ok();
  }
  case mambaserve::NixlControl::kPeersFinalized: {
    if (!cluster_)
      return Status::RuntimeError("NixlCluster missing");
    if (!ctrl.peers_finalized())
      return Status::Ok();
    std::lock_guard<std::mutex> lk(cluster_->mu);
    cluster_->peers_finalized = true;
    return Status::Ok();
  }
  // publish_md only travels worker -> parent.
  case mambaserve::NixlControl::kPublishMd:
  case mambaserve::NixlControl::BODY_NOT_SET:
    return Status::Ok();
  }
  return Status::Ok();
}

std::optional<mambaserve::TransportControl> NixlCommAgent::make_register_announce() {
  if (!cluster_)
    return std::nullopt;
  std::string md;
  {
    std::lock_guard<std::mutex> lk(cluster_->mu);
    auto it = cluster_->local_md.find(device_id());
    if (it == cluster_->local_md.end() || it->second.empty())
      return std::nullopt;
    md = it->second;
  }
  mambaserve::TransportControl msg;
  auto* pub = msg.mutable_nixl_ctrl()->mutable_publish_md();
  pub->set_device_id(device_id());
  pub->set_md(std::move(md));
  return msg;
}

void NixlCommAgent::shutdown() {
#if MAMBASERVE_WITH_NIXL
  for (auto& kv : xfers_) {
    if (kv.second.req) {
      agent_->releaseXferReq(kv.second.req);
      kv.second.req = nullptr;
    }
  }
  xfers_.clear();

  if (agent_ && slab_registered_) {
    nixl_reg_dlist_t dlist(VRAM_SEG);
    dlist.addDesc(slab_desc_);
    nixl_opt_args_t extra;
    if (backend_)
      extra.backends.push_back(backend_);
    agent_->deregisterMem(dlist, backend_ ? &extra : nullptr);
    slab_registered_ = false;
  }

  if (agent_ && cluster_) {
    std::lock_guard<std::mutex> lk(cluster_->mu);
    for (const auto& kv : cluster_->local_md) {
      if (kv.first == device_id())
        continue;
      agent_->invalidateRemoteMD(agent_name(kv.first));
    }
  }

  backend_ = nullptr;
  agent_.reset();
#else
  xfers_.clear();
#endif
  slab_ptr_ = nullptr;
  slab_bytes_ = 0;
  peers_loaded_ = false;
}
