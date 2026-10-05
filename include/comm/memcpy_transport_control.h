#pragma once

#include "comm/transport_control_plane.h"

#include <cstdint>
#include <mutex>
#include <unordered_map>

// Routes MemcpyPeer control traffic between the two workers of a migrate:
//   Recv announce (dst slot)  -> Send worker
//   Send done (seq_len, ok)   -> Recv worker
// (replaces MemcpyPeerCommAgent's process-static rendezvous map).
class MemcpyTransportControl final : public TransportControlPlane {
public:
  void on_upstream(size_t from_worker, const mambaserve::TransportControl& msg,
                   TransportSender& sender) override;
  void on_migrate_begin(uint64_t xfer_id, size_t src_idx, size_t dst_idx) override;
  void on_migrate_end(uint64_t xfer_id) override;

private:
  struct Route {
    size_t src_idx = 0;
    size_t dst_idx = 0;
  };

  std::mutex mu_;
  std::unordered_map<uint64_t, Route> routes_; // xfer_id -> {Send worker, Recv worker}
};
