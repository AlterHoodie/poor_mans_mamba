#pragma once

#include "comm/transport_control_plane.h"
#include "core/status.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Bootstraps the NCCL communicator over the control plane (replaces the shared
// NcclCluster): at cluster load it sends every worker a NcclBootstrap{nranks,
// rank, unique_id} and blocks until all ranks have acked ncclCommInitRank.
// There is no per-migrate traffic; NCCL matches Send/Recv itself.
class NcclTransportControl final : public TransportControlPlane {
public:
  // Generates a fresh ncclUniqueId. NotImplemented when built without NCCL.
  static StatusOr<std::unique_ptr<NcclTransportControl>> create(int nranks);

  // For tests: build with a caller-supplied id blob.
  NcclTransportControl(int nranks, std::string unique_id);

  // The bootstrap message for `rank` (also used by the benchmark, which has no scheduler).
  mambaserve::TransportControl make_bootstrap(int rank) const;

  void on_upstream(size_t from_worker, const mambaserve::TransportControl& msg,
                   TransportSender& sender) override;
  Status on_cluster_ready(TransportSender& sender) override;

  void set_ack_timeout(std::chrono::milliseconds t) { ack_timeout_ = t; }

private:
  int nranks_ = 0;
  std::string unique_id_;
  std::chrono::milliseconds ack_timeout_{60'000};

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<bool> acked_; // per rank
  int n_acked_ = 0;
  std::string error_; // first failure, empty if none
};
