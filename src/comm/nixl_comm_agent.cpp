#include "comm/nixl_comm_agent.h"

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

nixl_blob_t NixlCommAgent::pack_notif_(uint64_t xfer_id, int64_t seq_len) {
  nixl_blob_t blob(sizeof(uint64_t) + sizeof(int64_t), '\0');
  std::memcpy(blob.data(), &xfer_id, sizeof(uint64_t));
  std::memcpy(blob.data() + sizeof(uint64_t), &seq_len, sizeof(int64_t));
  return blob;
}

bool NixlCommAgent::unpack_notif_(const nixl_blob_t& blob, uint64_t* xfer_id, int64_t* seq_len) {
  if (blob.size() < sizeof(uint64_t) + sizeof(int64_t) || !xfer_id || !seq_len)
    return false;
  std::memcpy(xfer_id, blob.data(), sizeof(uint64_t));
  std::memcpy(seq_len, blob.data() + sizeof(uint64_t), sizeof(int64_t));
  return true;
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

StatusOr<XferHandle> NixlCommAgent::post(XferDesc& desc) {
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
  {
    std::lock_guard<std::mutex> lk(cluster_->mu);
    if (!cluster_->peers_finalized)
      return Status::RuntimeError("NIXL peers not finalized");
  }

  const auto local = static_cast<const char*>(desc.local_ptr);
  const auto base = static_cast<const char*>(slab_ptr_);
  if (local < base || local + desc.bytes > base + slab_bytes_)
    return Status::InvalidArgument("NIXL local_ptr outside registered slab");
  if (desc.peer_device_id < 0 || desc.peer_device_id == device_id())
    return Status::InvalidArgument("invalid peer_device_id for NIXL");

  // Recv's slot announce is sent to the Send peer via the scheduler's control
  // plane (see make_post_announce), not published here.

  const uint64_t hid = id_counter_.fetch_add(1, std::memory_order_relaxed);
  Entry e;
  e.role = desc.role;
  e.state = XferState::Pending;
  e.seq_len = desc.seq_len;
  e.xfer_id = desc.xfer_id;
  e.local_ptr = desc.local_ptr;
  e.bytes = desc.bytes;
  e.peer_device_id = desc.peer_device_id;
  xfers_[hid] = e;
  return XferHandle{hid};
#endif
}

#if MAMBASERVE_WITH_NIXL

XferState NixlCommAgent::poll_send_(Entry& e) {
  if (e.state != XferState::Pending)
    return e.state; // Done or Error

  if (e.posted) {
    // WRITE already posted; drive it to completion (this also sends the notif).
    nixl_status_t st = agent_->getXferStatus(e.req);
    if (st == NIXL_IN_PROG)
      return XferState::Pending;
    agent_->releaseXferReq(e.req);
    e.req = nullptr;
    e.state = (st == NIXL_SUCCESS) ? XferState::Done : XferState::Error;
    return e.state;
  }

  if (Status s = ensure_peers_loaded_(); !s.ok()) {
    e.state = XferState::Error;
    return e.state;
  }

  SlotAnnounce announce;
  {
    std::lock_guard<std::mutex> lk(announce_mu_);
    auto it = announces_.find(e.xfer_id);
    // wait for recv to publish destination slot ptr via slot announce
    if (it == announces_.end())
      return XferState::Pending;
    announce = it->second;
  }

  if (announce.bytes != e.bytes) {
    e.state = XferState::Error;
    return e.state;
  }

  if (cudaSetDevice(device_id()) != cudaSuccess) {
    e.state = XferState::Error;
    return e.state;
  }

  nixl_xfer_dlist_t src(VRAM_SEG);
  nixlBasicDesc src_desc(reinterpret_cast<uintptr_t>(e.local_ptr), e.bytes,
                         static_cast<uint64_t>(device_id()));
  src.addDesc(src_desc);

  nixl_xfer_dlist_t dst(VRAM_SEG);
  nixlBasicDesc dst_desc(reinterpret_cast<uintptr_t>(announce.dst_ptr), announce.bytes,
                         static_cast<uint64_t>(announce.dst_device));
  dst.addDesc(dst_desc);

  nixl_opt_args_t extra;
  extra.backends.push_back(backend_);
  extra.notif = pack_notif_(e.xfer_id, e.seq_len);

  const std::string remote = agent_name(e.peer_device_id);
  nixl_status_t st =
      agent_->createXferReq(NIXL_WRITE, src, dst, remote, e.req, &extra);
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
  {
    std::lock_guard<std::mutex> lk(announce_mu_);
    announces_.erase(e.xfer_id);
  }

  if (st == NIXL_SUCCESS) {
    agent_->releaseXferReq(e.req);
    e.req = nullptr;
    e.state = XferState::Done;
    return e.state;
  }
  // NIXL_IN_PROG: the UCX backend only flushes the WRITE and sends the
  // completion notif from getXferStatus(), so it must be polled until done.
  return XferState::Pending;
}

XferState NixlCommAgent::poll_recv_(Entry& e) {
  // recv polls just waits and listens to new completion notifs
  if (e.state != XferState::Pending)
    return e.state;

  nixl_notifs_t notifs;
  // drain new notifs
  nixl_status_t st = agent_->getNotifs(notifs);
  if (st != NIXL_SUCCESS) {
    e.state = XferState::Error;
    return e.state;
  }
  for (auto& kv : notifs) {
    for (auto& blob : kv.second)
      // keep track of pending notifs
      pending_notifs_.emplace_back(kv.first, std::move(blob));
  }

  const std::string peer = agent_name(e.peer_device_id);
  for (auto it = pending_notifs_.begin(); it != pending_notifs_.end(); ++it) {
    if (it->first != peer)
      continue;
    uint64_t xid = 0;
    int64_t sl = 0;
    // not completly transfered
    if (!unpack_notif_(it->second, &xid, &sl) || xid != e.xfer_id)
      continue;
    // if completely transfered
    e.seq_len = sl;
    e.state = XferState::Done;
    pending_notifs_.erase(it);
    return e.state;
  }
  return XferState::Pending;
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
  Entry& e = it->second;
  if (e.role == XferRole::Recv)
    return poll_recv_(e);
  return poll_send_(e);
#endif
}

int64_t NixlCommAgent::xfer_seq_len(const XferHandle& handle) {
  auto it = xfers_.find(handle.id);
  if (it == xfers_.end())
    return 0;
  return it->second.seq_len;
}

std::optional<mambaserve::TransportControl> NixlCommAgent::make_post_announce(const XferDesc& desc) {
  if (desc.role != XferRole::Recv)
    return std::nullopt;
  mambaserve::TransportControl msg;
  auto* a = msg.mutable_nixl_ctrl()->mutable_announce();
  a->set_xfer_id(desc.xfer_id);
  a->set_dst_ptr(reinterpret_cast<uint64_t>(desc.local_ptr));
  a->set_bytes(desc.bytes);
  a->set_dst_device(device_id());
  return msg;
}

Status NixlCommAgent::handle_transport(const mambaserve::TransportControl& msg) {
  if (msg.body_case() != mambaserve::TransportControl::kNixlCtrl)
    return Status::Ok();
  const mambaserve::NixlControl& ctrl = msg.nixl_ctrl();

  switch (ctrl.body_case()) {
  case mambaserve::NixlControl::kAnnounce: {
    const mambaserve::NixlSlotAnnounce& a = ctrl.announce();
    std::lock_guard<std::mutex> lk(announce_mu_);
    announces_[a.xfer_id()] = SlotAnnounce{
        .dst_ptr = reinterpret_cast<void*>(a.dst_ptr()),
        .bytes = static_cast<size_t>(a.bytes()),
        .dst_device = a.dst_device(),
    };
    return Status::Ok();
  }
  case mambaserve::NixlControl::kClear: {
    std::lock_guard<std::mutex> lk(announce_mu_);
    announces_.erase(ctrl.clear().xfer_id());
    return Status::Ok();
  }
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
  {
    std::lock_guard<std::mutex> lk(announce_mu_);
    announces_.clear();
  }
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

  pending_notifs_.clear();
  backend_ = nullptr;
  agent_.reset();
#else
  xfers_.clear();
#endif
  slab_ptr_ = nullptr;
  slab_bytes_ = 0;
  peers_loaded_ = false;
}
