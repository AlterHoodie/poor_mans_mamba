#pragma once

#include "core/status.h"
#include "proto/worker.pb.h"

#include <cstddef>
#include <cstdint>

// Narrow capability surface the cluster scheduler (owner of workers and
// their IPC channels) lends to a TransportControlPlane. The plane decides what
// to send and where; the sender only delivers.
struct TransportSender {
  virtual ~TransportSender() = default;

  virtual void send_transport(size_t worker_idx, const mambaserve::TransportControl& msg) = 0;
  virtual void broadcast_transport(const mambaserve::TransportControl& msg) = 0;
  virtual size_t n_workers() const = 0;
};

// Backend-specific control-plane brain (NIXL announce routing, NCCL bootstrap, ...).
// ClusterScheduler only forwards transport Envelopes and migrate lifecycle
// hooks to it, so it never has to know about a backend's protocol.
//
// on_upstream may be called from the scheduler's ingress thread while
// on_migrate_* are called from the event/caller threads; implementations must
// be thread-safe.
class TransportControlPlane {
public:
  virtual ~TransportControlPlane() = default;

  // A transport message arrived from `from_worker`.
  virtual void on_upstream(size_t from_worker, const mambaserve::TransportControl& msg,
                           TransportSender& sender) = 0;

  // All workers exist and the scheduler's ingress is live; the plane may push
  // bootstrap traffic via the sender and block until workers ack.
  // A non-OK status fails ClusterScheduler::start.
  virtual Status on_cluster_ready(TransportSender& /*sender*/) { return Status::Ok(); }
};

// For backends with no parent-routed control traffic (fallback / tests).
class NoopTransportControlPlane final : public TransportControlPlane {
public:
  void on_upstream(size_t, const mambaserve::TransportControl&, TransportSender&) override {}
};
