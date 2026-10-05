#include "comm/nccl_transport_control.h"

#include "telemetry/log.h"

#include <utility>

#if !defined(MAMBASERVE_WITH_NCCL)
#define MAMBASERVE_WITH_NCCL 0
#endif

#if MAMBASERVE_WITH_NCCL
#include <nccl.h>
#endif

StatusOr<std::unique_ptr<NcclTransportControl>> NcclTransportControl::create(int nranks) {
  if (nranks <= 0)
    return Status::InvalidArgument("nranks must be > 0");
#if MAMBASERVE_WITH_NCCL
  ncclUniqueId id;
  ncclResult_t r = ncclGetUniqueId(&id);
  if (r != ncclSuccess)
    return Status::RuntimeError(std::string("ncclGetUniqueId: ") + ncclGetErrorString(r));
  return std::make_unique<NcclTransportControl>(
      nranks, std::string(id.internal, sizeof(id.internal)));
#else
  return Status::NotImplemented("NCCL transport not enabled (build with MAMBASERVE_WITH_NCCL=ON)");
#endif
}

NcclTransportControl::NcclTransportControl(int nranks, std::string unique_id)
    : nranks_(nranks), unique_id_(std::move(unique_id)),
      acked_(static_cast<size_t>(nranks > 0 ? nranks : 0), false) {}

mambaserve::TransportControl NcclTransportControl::make_bootstrap(int rank) const {
  mambaserve::TransportControl msg;
  auto* boot = msg.mutable_nccl_ctrl()->mutable_bootstrap();
  boot->set_nranks(nranks_);
  boot->set_rank(rank);
  boot->set_unique_id(unique_id_);
  return msg;
}

void NcclTransportControl::on_upstream(size_t from_worker, const mambaserve::TransportControl& msg,
                                       TransportSender& /*sender*/) {
  if (msg.body_case() != mambaserve::TransportControl::kNcclCtrl)
    return;
  const mambaserve::NcclControl& ctrl = msg.nccl_ctrl();
  if (ctrl.body_case() != mambaserve::NcclControl::kBootstrapAck)
    return;
  const mambaserve::NcclBootstrapAck& ack = ctrl.bootstrap_ack();

  {
    std::lock_guard<std::mutex> lk(mu_);
    const int rank = ack.rank();
    if (!ack.ok()) {
      if (error_.empty())
        error_ = "NCCL bootstrap failed on rank " + std::to_string(rank) + ": " + ack.error();
    } else if (rank < 0 || rank >= nranks_) {
      if (error_.empty())
        error_ = "NCCL bootstrap ack from unexpected rank " + std::to_string(rank);
    } else if (!acked_[static_cast<size_t>(rank)]) {
      acked_[static_cast<size_t>(rank)] = true;
      ++n_acked_;
    }
    LOG_DEBUG("nccl bootstrap ack from worker %zu rank=%d ok=%d (%d/%d)", from_worker, rank,
              ack.ok() ? 1 : 0, n_acked_, nranks_);
  }
  cv_.notify_all();
}

Status NcclTransportControl::on_cluster_ready(TransportSender& sender) {
  if (static_cast<int>(sender.n_workers()) != nranks_)
    return Status::InvalidArgument("NCCL nranks does not match worker count");

  {
    std::lock_guard<std::mutex> lk(mu_);
    acked_.assign(static_cast<size_t>(nranks_), false);
    n_acked_ = 0;
    error_.clear();
  }

  // Worker i owns rank i (rank == device id).
  for (int i = 0; i < nranks_; ++i)
    sender.send_transport(static_cast<size_t>(i), make_bootstrap(i));

  std::unique_lock<std::mutex> lk(mu_);
  const bool settled =
      cv_.wait_for(lk, ack_timeout_, [&] { return !error_.empty() || n_acked_ == nranks_; });
  if (!error_.empty())
    return Status::RuntimeError(error_);
  if (!settled)
    return Status::RuntimeError("timed out waiting for NCCL bootstrap acks");
  return Status::Ok();
}
