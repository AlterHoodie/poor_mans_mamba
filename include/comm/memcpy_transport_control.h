#pragma once

#include "comm/transport_control_plane.h"

// MemcpyPeer is receiver-driven (src_ptr on MigrateCmd); no parent-routed
// per-xfer control traffic.
class MemcpyTransportControl final : public TransportControlPlane {
public:
  void on_upstream(size_t, const mambaserve::TransportControl&, TransportSender&) override {}
};
