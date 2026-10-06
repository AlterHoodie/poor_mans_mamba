#include "comm/nixl_transport_control.h"

#include <string>
#include <utility>
#include <vector>

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

void NixlTransportControl::on_upstream(size_t /*from_worker*/,
                                       const mambaserve::TransportControl& msg,
                                       TransportSender& /*sender*/) {
  if (msg.body_case() != mambaserve::TransportControl::kNixlCtrl)
    return;
  const mambaserve::NixlControl& ctrl = msg.nixl_ctrl();
  if (ctrl.body_case() != mambaserve::NixlControl::kPublishMd)
    return;
  {
    std::lock_guard<std::mutex> lk(mu_);
    md_[ctrl.publish_md().device_id()] = ctrl.publish_md().md();
  }
  md_cv_.notify_all();
}
