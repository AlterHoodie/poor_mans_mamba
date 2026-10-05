#pragma once

#include "comm/transport_control_plane.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

// NIXL control plane. Everything NIXL workers used to share in-process goes through
// the parent so workers can be threads or separate processes:
//  - Slot announces: Recv publishes its destination slot up to the parent, which
//    forwards it to the Send side of the migrate.
//  - Metadata exchange: every worker publishes its agent metadata (publish_md) once
//    it is Ready. on_cluster_ready() waits for all of them, fans each worker the
//    others' metadata (peer_md) and then broadcasts peers_finalized.
class NixlTransportControl final : public TransportControlPlane {
public:
  explicit NixlTransportControl(std::chrono::milliseconds md_timeout = std::chrono::seconds(60))
      : md_timeout_(md_timeout) {}

  void on_upstream(size_t from_worker, const mambaserve::TransportControl& msg,
                   TransportSender& sender) override;
  Status on_cluster_ready(TransportSender& sender) override;
  void on_migrate_begin(uint64_t xfer_id, size_t src_idx, size_t dst_idx) override;
  void on_migrate_end(uint64_t xfer_id) override;

private:
  struct Route {
    size_t src_idx = 0;
    size_t dst_idx = 0;
  };

  const std::chrono::milliseconds md_timeout_;

  std::mutex mu_;
  std::condition_variable md_cv_;
  std::unordered_map<uint64_t, Route> routes_; // xfer_id -> {Send worker, Recv worker}
  std::unordered_map<int, std::string> md_;    // device_id -> published agent metadata
};
