#pragma once

#include "comm/transport_control_plane.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

// NIXL control plane: agent metadata exchange only. Per-xfer slot announces are
// gone (Recv drives NIXL_READ with src_ptr from MigrateCmd). Every worker
// publishes getLocalMD (publish_md) once Ready; on_cluster_ready() waits, fans
// peer_md to each worker, then broadcasts peers_finalized.
class NixlTransportControl final : public TransportControlPlane {
public:
  explicit NixlTransportControl(std::chrono::milliseconds md_timeout = std::chrono::seconds(60))
      : md_timeout_(md_timeout) {}

  void on_upstream(size_t from_worker, const mambaserve::TransportControl& msg,
                   TransportSender& sender) override;
  Status on_cluster_ready(TransportSender& sender) override;

private:
  const std::chrono::milliseconds md_timeout_;

  std::mutex mu_;
  std::condition_variable md_cv_;
  std::unordered_map<int, std::string> md_; // device_id -> published agent metadata
};
