#include "comm/memcpy_transport_control.h"

#include "telemetry/log.h"

#include <optional>

void MemcpyTransportControl::on_migrate_begin(uint64_t xfer_id, size_t src_idx, size_t dst_idx) {
  // create route for xfer_id
  std::lock_guard<std::mutex> lk(mu_);
  routes_[xfer_id] = Route{.src_idx = src_idx, .dst_idx = dst_idx};
}

void MemcpyTransportControl::on_migrate_end(uint64_t xfer_id) {
  // erase route for xfer_id
  std::lock_guard<std::mutex> lk(mu_);
  routes_.erase(xfer_id);
}

void MemcpyTransportControl::on_upstream(size_t from_worker,
                                         const mambaserve::TransportControl& msg,
                                         TransportSender& sender) {
  if (msg.body_case() != mambaserve::TransportControl::kMemcpyCtrl)
    return;
  const mambaserve::MemcpyControl& ctrl = msg.memcpy_ctrl();

  uint64_t xfer_id = 0;
  switch (ctrl.body_case()) {
  case mambaserve::MemcpyControl::kAnnounce:
    xfer_id = ctrl.announce().xfer_id();
    break;
  case mambaserve::MemcpyControl::kClear:
    xfer_id = ctrl.clear().xfer_id();
    break;
  case mambaserve::MemcpyControl::kDone:
    xfer_id = ctrl.done().xfer_id();
    break;
  case mambaserve::MemcpyControl::BODY_NOT_SET:
    return;
  }

  // Forward to the other side of the migrate.
  std::optional<size_t> target;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = routes_.find(xfer_id);
    if (it == routes_.end()) {
      LOG_DEBUG("memcpy control for unknown xfer_id=%llu from worker %zu dropped",
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
