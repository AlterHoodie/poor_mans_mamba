#include "comm/nixl_transport_control.h"

#include "telemetry/log.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

void NixlTransportControl::on_migrate_begin(uint64_t xfer_id, size_t src_idx, size_t dst_idx) {
  std::lock_guard<std::mutex> lk(mu_);
  routes_[xfer_id] = Route{.src_idx = src_idx, .dst_idx = dst_idx};
}

void NixlTransportControl::on_migrate_end(uint64_t xfer_id) {
  std::lock_guard<std::mutex> lk(mu_);
  routes_.erase(xfer_id);
}

Status NixlTransportControl::on_cluster_ready(TransportSender& sender) {
  const size_t n = sender.n_workers();

  std::vector<std::pair<int, std::string>> all;
  {
    std::unique_lock<std::mutex> lk(mu_);
    const bool complete = md_cv_.wait_for(lk, md_timeout_, [&]() { return md_.size() >= n; });
    if (!complete)
      return Status::RuntimeError("timed out waiting for NIXL metadata from all workers");
    all.assign(md_.begin(), md_.end());
  }

  // Worker index == device id, so device_id doubles as the delivery target.
  for (size_t target = 0; target < n; ++target) {
    for (const auto& [device_id, md] : all) {
      if (static_cast<size_t>(device_id) == target)
        continue;
      mambaserve::TransportControl msg;
      auto* peer = msg.mutable_nixl_ctrl()->mutable_peer_md();
      peer->set_device_id(device_id);
      peer->set_md(md);
      sender.send_transport(target, msg);
    }
  }

  mambaserve::TransportControl done;
  done.mutable_nixl_ctrl()->set_peers_finalized(true);
  sender.broadcast_transport(done);
  return Status::Ok();
}

void NixlTransportControl::on_upstream(size_t from_worker, const mambaserve::TransportControl& msg,
                                       TransportSender& sender) {
  if (msg.body_case() != mambaserve::TransportControl::kNixlCtrl)
    return;
  const mambaserve::NixlControl& ctrl = msg.nixl_ctrl();

  uint64_t xfer_id = 0;
  switch (ctrl.body_case()) {
  case mambaserve::NixlControl::kAnnounce:
    xfer_id = ctrl.announce().xfer_id();
    break;
  case mambaserve::NixlControl::kClear:
    xfer_id = ctrl.clear().xfer_id();
    break;
  case mambaserve::NixlControl::kPublishMd: {
    {
      std::lock_guard<std::mutex> lk(mu_);
      md_[ctrl.publish_md().device_id()] = ctrl.publish_md().md();
    }
    md_cv_.notify_all();
    return;
  }
  // peer_md / peers_finalized only travel parent -> worker.
  case mambaserve::NixlControl::kPeerMd:
  case mambaserve::NixlControl::kPeersFinalized:
  case mambaserve::NixlControl::BODY_NOT_SET:
    return;
  }

  // Forward to the other side of the migrate.
  std::optional<size_t> target;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = routes_.find(xfer_id);
    if (it == routes_.end()) {
      LOG_DEBUG("nixl control for unknown xfer_id=%llu from worker %zu dropped",
                static_cast<unsigned long long>(xfer_id), from_worker);
      return;
    }
    const Route& r = it->second;
    if (from_worker == r.dst_idx)
      target = r.src_idx;
    else if (from_worker == r.src_idx)
      target = r.dst_idx;
  }
  if (target)
    sender.send_transport(*target, msg);
}
